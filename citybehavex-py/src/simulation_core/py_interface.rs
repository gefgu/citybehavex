use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::simulation_core::engine::simulate;
use crate::simulation_core::inputs::{
    ActivityInputs, CoreInputs, DiaryInputs, InitialLocationInputs, LocationInputs,
    SimulationParams, SocialGraphInputs,
};

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyo3(signature = (
    latitudes, longitudes, relevances, distances,
    neighbor_starts, neighbors,
    diary_timestamps, diary_abs_locs, diary_starts, diary_ends,
    rho, gamma, alpha,
    start_ts, end_ts, indipendency_window_s, dt_update_mob_sim_s,
    slot_seconds, car_speed_kmh,
    n_agents, master_seed=None, starting_locs=None,
    starting_locs_mode_relevance=false,
    work_tiles=None,
    edge_profile_sim=None,
    act_embs=None, act_dur_mu=None, act_dur_sigma=None,
    purpose_act_starts=None, purpose_acts=None,
    profile_embs=None, emb_dim=0usize,
    act_kappa=1.0f64, act_temp=0.5f64
))]
pub fn simulation_core_simulate_agents<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    relevances: PyReadonlyArray1<'py, f64>,
    distances: PyReadonlyArray1<'py, f64>,
    neighbor_starts: PyReadonlyArray1<'py, i64>,
    neighbors: PyReadonlyArray1<'py, i64>,
    diary_timestamps: PyReadonlyArray1<'py, i64>,
    diary_abs_locs: PyReadonlyArray1<'py, i32>,
    diary_starts: PyReadonlyArray1<'py, i64>,
    diary_ends: PyReadonlyArray1<'py, i64>,
    rho: f64,
    gamma: f64,
    alpha: f64,
    start_ts: i64,
    end_ts: i64,
    indipendency_window_s: i64,
    dt_update_mob_sim_s: i64,
    slot_seconds: i64,
    car_speed_kmh: f64,
    n_agents: usize,
    master_seed: Option<u64>,
    starting_locs: Option<PyReadonlyArray1<'py, i64>>,
    starting_locs_mode_relevance: bool,
    work_tiles: Option<PyReadonlyArray1<'py, i64>>,
    edge_profile_sim: Option<PyReadonlyArray1<'py, f64>>,
    act_embs: Option<PyReadonlyArray1<'py, f64>>,
    act_dur_mu: Option<PyReadonlyArray1<'py, f64>>,
    act_dur_sigma: Option<PyReadonlyArray1<'py, f64>>,
    purpose_act_starts: Option<PyReadonlyArray1<'py, i64>>,
    purpose_acts: Option<PyReadonlyArray1<'py, i64>>,
    profile_embs: Option<PyReadonlyArray1<'py, f64>>,
    emb_dim: usize,
    act_kappa: f64,
    act_temp: f64,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let rels = relevances.as_slice()?;
    let dists = distances.as_slice()?;
    let ns_raw = neighbor_starts.as_slice()?;
    let nb_raw = neighbors.as_slice()?;
    let dt_raw = diary_timestamps.as_slice()?;
    let da_raw = diary_abs_locs.as_slice()?;
    let ds_raw = diary_starts.as_slice()?;
    let de_raw = diary_ends.as_slice()?;

    let ns: Vec<usize> = ns_raw.iter().map(|&v| v.max(0) as usize).collect();
    let nb: Vec<usize> = nb_raw.iter().map(|&v| v.max(0) as usize).collect();
    let ds: Vec<usize> = ds_raw.iter().map(|&v| v.max(0) as usize).collect();
    let de: Vec<usize> = de_raw.iter().map(|&v| v.max(0) as usize).collect();

    let sl_buf: Vec<usize>;
    let sl: Option<&[usize]> = match &starting_locs {
        Some(v) => {
            sl_buf = v.as_slice()?.iter().map(|&x| x.max(0) as usize).collect();
            Some(&sl_buf)
        }
        None => None,
    };

    let wt_buf: Vec<usize>;
    let wt_empty: &[usize] = &[];
    let wt: &[usize] = match &work_tiles {
        Some(v) => {
            wt_buf = v.as_slice()?.iter().map(|&x| x.max(0) as usize).collect();
            &wt_buf
        }
        None => wt_empty,
    };

    let eps_buf: Vec<f64>;
    let eps_empty: &[f64] = &[];
    let eps: &[f64] = match &edge_profile_sim {
        Some(v) => {
            eps_buf = v.as_slice()?.to_vec();
            &eps_buf
        }
        None => eps_empty,
    };

    let act_embs_empty: &[f64] = &[];
    let act_embs_s = match &act_embs {
        Some(v) => v.as_slice()?,
        None => act_embs_empty,
    };
    let act_dur_mu_empty: &[f64] = &[];
    let act_dur_mu_s = match &act_dur_mu {
        Some(v) => v.as_slice()?,
        None => act_dur_mu_empty,
    };
    let act_dur_sigma_empty: &[f64] = &[];
    let act_dur_sigma_s = match &act_dur_sigma {
        Some(v) => v.as_slice()?,
        None => act_dur_sigma_empty,
    };
    let profile_embs_empty: &[f64] = &[];
    let profile_embs_s = match &profile_embs {
        Some(v) => v.as_slice()?,
        None => profile_embs_empty,
    };
    let purpose_act_starts_v: Vec<usize> = match &purpose_act_starts {
        Some(v) => v.as_slice()?.iter().map(|&x| x.max(0) as usize).collect(),
        None => Vec::new(),
    };
    let purpose_acts_v: Vec<usize> = match &purpose_acts {
        Some(v) => v.as_slice()?.iter().map(|&x| x.max(0) as usize).collect(),
        None => Vec::new(),
    };

    let output = simulate(CoreInputs {
        locations: LocationInputs {
            lats,
            lngs,
            relevances: rels,
            distances: dists,
        },
        social_graph: SocialGraphInputs {
            neighbor_starts: &ns,
            neighbors: &nb,
            edge_profile_sim: eps,
        },
        diary: DiaryInputs {
            timestamps: dt_raw,
            abstract_locations: da_raw,
            starts: &ds,
            ends: &de,
        },
        params: SimulationParams {
            rho,
            gamma,
            alpha,
            start_ts,
            end_ts,
            indipendency_window_s,
            dt_update_mob_sim_s,
            slot_seconds,
            car_speed_kmh,
            n_agents,
            master_seed,
        },
        initial_locations: InitialLocationInputs {
            starting_locs: sl,
            starting_locs_mode_relevance,
            work_tiles: wt,
        },
        activities: ActivityInputs {
            act_embs: act_embs_s,
            act_dur_mu: act_dur_mu_s,
            act_dur_sigma: act_dur_sigma_s,
            purpose_act_starts: &purpose_act_starts_v,
            purpose_acts: &purpose_acts_v,
            profile_embs: profile_embs_s,
            emb_dim,
            kappa: act_kappa,
            temperature: act_temp,
        },
    })
    .map_err(PyValueError::new_err)?;

    Ok((
        output.agents.into_pyarray(py),
        output.lats.into_pyarray(py),
        output.lngs.into_pyarray(py),
        output.arrival.into_pyarray(py),
        output.departure.into_pyarray(py),
        output.duration.into_pyarray(py),
        output.encounter_agent.into_pyarray(py),
        output.encounter_contact.into_pyarray(py),
        output.encounter_tile.into_pyarray(py),
        output.encounter_ts.into_pyarray(py),
        output.activity.into_pyarray(py),
    ))
}
