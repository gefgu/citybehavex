use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    window::PresentMode,
};

// --- COMPONENTS & RESOURCES ---

#[derive(Component)]
struct FpsText;

#[derive(Component)]
struct TimelineText;

// 1. A new Resource to track our custom time
#[derive(Resource)]
struct Timeline {
    current_time: f32,     // 0 to 240 seconds (4 minutes)
    speed_multiplier: f32, // e.g., 1.0 (normal), 5.0, 15.0, or -5.0 (rewind)
}

// 2. We don't need direction/speed in the component anymore, 
// just the side length to calculate the path.
#[derive(Component)]
struct BoxPath {
    side_length: f32, 
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.15)))
        // Initialize our Timeline resource starting at 0 seconds, normal speed
        .insert_resource(Timeline {
            current_time: 0.0,
            speed_multiplier: 1.0,
        })
        .add_plugins(DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "My Bevy Box - Timeline Edition".into(),
                    present_mode: PresentMode::Fifo, 
                    ..default()
                }),
                ..default()
            })
            .disable::<bevy::audio::AudioPlugin>()
        )
        .add_plugins(FrameTimeDiagnosticsPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (
            move_camera, 
            update_ui_text, 
            toggle_fps_visibility, 
            control_timeline, // Handles changing speed/rewinding
            animate_box       // Updated time-based animation
        ))
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());

    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::srgb(0.8, 0.3, 0.3),
                custom_size: Some(Vec2::new(150.0, 150.0)),
                ..default()
            },
            ..default()
        },
        BoxPath {
            side_length: 300.0, 
        }
    ));

    // UI Container
    commands.spawn(NodeBundle {
        style: Style {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            flex_direction: FlexDirection::Column,
            ..default()
        },
        ..default()
    }).with_children(|parent| {
        // FPS Text
        parent.spawn((
            TextBundle {
                text: Text::from_section(
                    "FPS: --",
                    TextStyle { font_size: 24.0, color: Color::WHITE, ..default() },
                ),
                visibility: Visibility::Hidden, // Set it directly inside the bundle
                ..default()
            },
            FpsText,
        ));
        
        // Timeline Info Text
        parent.spawn((
            TextBundle::from_section(
                "Time: 0.00s | Speed: 1x",
                TextStyle { font_size: 24.0, color: Color::srgb(0.5, 0.8, 1.0), ..default() },
            ),
            TimelineText,
        ));
    });
}

// 3. THE TIMELINE CONTROLLER
fn control_timeline(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut timeline: ResMut<Timeline>,
) {
    // Change speeds using number keys
    if keyboard_input.just_pressed(KeyCode::Digit1) {
        timeline.speed_multiplier = 1.0;
    }
    if keyboard_input.just_pressed(KeyCode::Digit2) {
        timeline.speed_multiplier = 5.0; // 5 seconds per second
    }
    if keyboard_input.just_pressed(KeyCode::Digit3) {
        timeline.speed_multiplier = 15.0; // 15 seconds per second
    }
    
    // Pause
    if keyboard_input.just_pressed(KeyCode::Space) {
        timeline.speed_multiplier = 0.0;
    }

    // Toggle Reverse/Rewind
    if keyboard_input.just_pressed(KeyCode::KeyR) {
        if timeline.speed_multiplier == 0.0 {
            timeline.speed_multiplier = -1.0; // Unpause into reverse
        } else {
            timeline.speed_multiplier *= -1.0; // Flip current direction
        }
    }
}

// 4. THE ANIMATION SYSTEM (Now strictly Time-Based)
fn animate_box(
    mut timeline: ResMut<Timeline>,
    time: Res<Time>,
    mut query: Query<(&mut Transform, &BoxPath)>,
) {
    // Advance the timeline by the real-world delta, multiplied by our custom speed
    timeline.current_time += time.delta_seconds() * timeline.speed_multiplier;
    
    // Total duration is 240 seconds (4 minutes). 
    // rem_euclid keeps the time cleanly looping between 0.0 and 239.999, even when rewinding (negative time)
    let t = timeline.current_time.rem_euclid(240.0);

    for (mut transform, box_path) in &mut query {
        let length = box_path.side_length;
        
        let mut x = 0.0;
        let mut y = 0.0;
        
        // Map our 240 second timeline to the 4 sides of the square
        if t < 60.0 {
            // Corner 0 -> 1 (Moving Up)
            let progress = t / 60.0; 
            x = 0.0;
            y = length * progress;
        } else if t < 120.0 {
            // Corner 1 -> 2 (Moving Right)
            let progress = (t - 60.0) / 60.0;
            x = length * progress;
            y = length;
        } else if t < 180.0 {
            // Corner 2 -> 3 (Moving Down)
            let progress = (t - 120.0) / 60.0;
            x = length;
            y = length - (length * progress);
        } else {
            // Corner 3 -> 0 (Moving Left)
            let progress = (t - 180.0) / 60.0;
            x = length - (length * progress);
            y = 0.0;
        }
        
        // Apply the calculated position
        transform.translation.x = x;
        transform.translation.y = y;
    }
}

// --- UPDATED UI LOGIC ---

fn update_ui_text(
    diagnostics: Res<DiagnosticsStore>,
    timeline: Res<Timeline>,
    mut fps_query: Query<&mut Text, (With<FpsText>, Without<TimelineText>)>,
    mut timeline_query: Query<&mut Text, (With<TimelineText>, Without<FpsText>)>,
) {
    // Update FPS
    for mut text in &mut fps_query {
        if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                text.sections[0].value = format!("FPS: {:.0}", value);
            }
        }
    }

    // Update Timeline Info
    for mut text in &mut timeline_query {
        let t = timeline.current_time.rem_euclid(240.0);
        
        // Determine which corner/side we are currently on
        let current_side = if t < 60.0 { "Side 1 (Up)" } 
            else if t < 120.0 { "Side 2 (Right)" } 
            else if t < 180.0 { "Side 3 (Down)" } 
            else { "Side 4 (Left)" };

        text.sections[0].value = format!(
            "Time: {:0>5.1}s | Speed: {:>3}x | {}", 
            t, timeline.speed_multiplier, current_side
        );
    }
}

// --- UNCHANGED CODE ---

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

fn move_camera(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut OrthographicProjection), With<Camera>>,
) {
    let move_speed = 400.0;
    let zoom_speed = 1.5;
    let mut direction = Vec3::ZERO;
    let mut zoom_delta = 0.0;

    if keyboard_input.pressed(KeyCode::KeyW) || keyboard_input.pressed(KeyCode::ArrowUp) { direction.y += 1.0; }
    if keyboard_input.pressed(KeyCode::KeyS) || keyboard_input.pressed(KeyCode::ArrowDown) { direction.y -= 1.0; }
    if keyboard_input.pressed(KeyCode::KeyA) || keyboard_input.pressed(KeyCode::ArrowLeft) { direction.x -= 1.0; }
    if keyboard_input.pressed(KeyCode::KeyD) || keyboard_input.pressed(KeyCode::ArrowRight) { direction.x += 1.0; }

    if keyboard_input.pressed(KeyCode::KeyQ) { zoom_delta -= 1.0; }
    if keyboard_input.pressed(KeyCode::KeyE) { zoom_delta += 1.0; }

    if direction.length() > 0.0 { direction = direction.normalize(); }

    for (mut transform, mut projection) in &mut query {
        transform.translation += direction * move_speed * projection.scale * time.delta_seconds();
        projection.scale += zoom_delta * zoom_speed * time.delta_seconds();
        projection.scale = projection.scale.max(0.1);
    }
}