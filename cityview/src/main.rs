use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    render::{mesh::PrimitiveTopology, render_asset::RenderAssetUsages},
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
    window::PresentMode,
};
use flatgeobuf::*;
use geozero::ToGeo;
use std::fs::File;
use std::io::BufReader;

// --- COMPONENTS & RESOURCES ---

#[derive(Component)]
struct FpsText;

/// Path to the FlatGeobuf map file, taken from the first CLI argument.
#[derive(Resource)]
struct MapPath(Option<String>);

/// A single shape read from the file: a triangle soup (every 3 vertices = 1 triangle),
/// already in EPSG:3857 metres.
struct Shape {
    verts: Vec<[f64; 2]>,
}

fn main() {
    let map_path = std::env::args().nth(1);

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.15)))
        .insert_resource(MapPath(map_path))
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "CityView".into(),
                        present_mode: PresentMode::Fifo,
                        ..default()
                    }),
                    ..default()
                })
                .disable::<bevy::audio::AudioPlugin>(),
        )
        .add_plugins(FrameTimeDiagnosticsPlugin)
        .add_systems(Startup, (setup, load_map))
        .add_systems(Update, (move_camera, update_fps_text, toggle_fps_visibility))
        .run();
}

fn setup(mut commands: Commands) {
    // Zoom out a little so a ~1.7 km bbox fits in the default window.
    let mut camera = Camera2dBundle::default();
    camera.projection.scale = 2.0;
    commands.spawn(camera);

    commands
        .spawn(NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                left: Val::Px(10.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((
                TextBundle {
                    text: Text::from_section(
                        "FPS: --",
                        TextStyle {
                            font_size: 24.0,
                            color: Color::WHITE,
                            ..default()
                        },
                    ),
                    visibility: Visibility::Hidden,
                    ..default()
                },
                FpsText,
            ));
        });
}

// --- MAP LOADING ---

fn load_map(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    map_path: Res<MapPath>,
) {
    let Some(path) = map_path.0.as_ref() else {
        warn!("No map file provided. Usage: cityview <path.fgb>");
        return;
    };

    let shapes = match read_shapes(path) {
        Ok(shapes) => shapes,
        Err(err) => {
            error!("Failed to load map '{path}': {err}");
            return;
        }
    };

    if shapes.is_empty() {
        warn!("Map file '{path}' contained no renderable geometry.");
        return;
    }

    // Center the geometry on the origin (3857 metres are huge; centering keeps f32 precise
    // and places the map under the default camera).
    let (mut min_x, mut min_y) = (f64::MAX, f64::MAX);
    let (mut max_x, mut max_y) = (f64::MIN, f64::MIN);
    for shape in &shapes {
        for v in &shape.verts {
            min_x = min_x.min(v[0]);
            min_y = min_y.min(v[1]);
            max_x = max_x.max(v[0]);
            max_y = max_y.max(v[1]);
        }
    }
    let center_x = (min_x + max_x) / 2.0;
    let center_y = (min_y + max_y) / 2.0;

    for (i, shape) in shapes.iter().enumerate() {
        let positions: Vec<[f32; 3]> = shape
            .verts
            .iter()
            .map(|v| [(v[0] - center_x) as f32, (v[1] - center_y) as f32, 0.0])
            .collect();

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);

        // A distinct hue per shape (golden-angle spacing avoids adjacent shapes clashing).
        let color = Color::hsl((i as f32 * 137.5).rem_euclid(360.0), 0.6, 0.55);

        commands.spawn(MaterialMesh2dBundle {
            mesh: Mesh2dHandle(meshes.add(mesh)),
            material: materials.add(color),
            ..default()
        });
    }

    info!("Loaded {} shapes from '{}'.", shapes.len(), path);
}

/// Read every feature from the FlatGeobuf file as a triangle soup. The exporter stores each
/// shape as a MultiPolygon of triangles, so each polygon part contributes one triangle.
fn read_shapes(path: &str) -> std::result::Result<Vec<Shape>, Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut fgb = FgbReader::open(&mut reader)?.select_all()?;

    let mut shapes = Vec::new();
    while let Some(feature) = fgb.next()? {
        let geom = feature.to_geo()?;
        let mut verts = Vec::new();
        collect_triangles(&geom, &mut verts);
        if !verts.is_empty() {
            shapes.push(Shape { verts });
        }
    }
    Ok(shapes)
}

fn collect_triangles(geom: &geo_types::Geometry<f64>, out: &mut Vec<[f64; 2]>) {
    use geo_types::Geometry;
    match geom {
        Geometry::MultiPolygon(mp) => {
            for poly in &mp.0 {
                push_triangle(poly, out);
            }
        }
        Geometry::Polygon(poly) => push_triangle(poly, out),
        Geometry::GeometryCollection(gc) => {
            for g in &gc.0 {
                collect_triangles(g, out);
            }
        }
        _ => {}
    }
}

fn push_triangle(poly: &geo_types::Polygon<f64>, out: &mut Vec<[f64; 2]>) {
    // Triangle polygons have a closed 4-point exterior ring; the first 3 points are the triangle.
    let ring = poly.exterior();
    if ring.0.len() >= 3 {
        for c in ring.0.iter().take(3) {
            out.push([c.x, c.y]);
        }
    }
}

// --- CAMERA (pan + zoom) ---

fn move_camera(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut OrthographicProjection), With<Camera>>,
) {
    let move_speed = 400.0;
    let zoom_speed = 1.5;
    let mut direction = Vec3::ZERO;
    let mut zoom_delta = 0.0;

    if keyboard_input.pressed(KeyCode::KeyW) || keyboard_input.pressed(KeyCode::ArrowUp) {
        direction.y += 1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyS) || keyboard_input.pressed(KeyCode::ArrowDown) {
        direction.y -= 1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyA) || keyboard_input.pressed(KeyCode::ArrowLeft) {
        direction.x -= 1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyD) || keyboard_input.pressed(KeyCode::ArrowRight) {
        direction.x += 1.0;
    }

    if keyboard_input.pressed(KeyCode::KeyQ) {
        zoom_delta -= 1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyE) {
        zoom_delta += 1.0;
    }

    if direction.length() > 0.0 {
        direction = direction.normalize();
    }

    for (mut transform, mut projection) in &mut query {
        transform.translation += direction * move_speed * projection.scale * time.delta_seconds();
        projection.scale += zoom_delta * zoom_speed * time.delta_seconds();
        projection.scale = projection.scale.max(0.1);
    }
}

// --- UI ---

fn update_fps_text(
    diagnostics: Res<DiagnosticsStore>,
    mut fps_query: Query<&mut Text, With<FpsText>>,
) {
    for mut text in &mut fps_query {
        if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                text.sections[0].value = format!("FPS: {:.0}", value);
            }
        }
    }
}

fn toggle_fps_visibility(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut Visibility, With<FpsText>>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyF) {
        for mut visibility in &mut query {
            *visibility = match *visibility {
                Visibility::Hidden => Visibility::Visible,
                _ => Visibility::Hidden,
            };
        }
    }
}
