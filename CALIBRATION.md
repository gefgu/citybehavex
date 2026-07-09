# Calibration Guide

This guide explains the validation metrics shown in the dashboard and lists
practical ways to improve them. Treat every suggestion as a hypothesis to test:
change one or two knobs, rerun the simulation, then compare the same filters and
run length.

## Distribution and Wasserstein Metrics

Lower Wasserstein values are better: they mean the synthetic distribution is
closer to the observed distribution in the metric's native unit.

| Metric | What it represents | Calibration levers |
| --- | --- | --- |
| Jump lengths | Trip-to-trip movement distance. | Tune `simulation.gravity_deterrence_exponent` for distance decay, `simulation.rho` for exploration, `simulation.gamma` for preferential return, and `comparison.road_network_distance`/`road_network.enabled` when road distance should replace straight-line distance. Fine-tune the schedule aligner if diaries choose too many far or local stops for a profile. |
| Visits per user | Number of visits or stays generated per user. | Tune `diaries.location_count_mu`, `diaries.location_count_sigma`, `diaries.max_locations`, `schedule.alpha_beta_a`, `schedule.alpha_beta_b`, and `simulation.gamma`. These control daily stop complexity and reuse. |
| Radius of gyration | Spatial spread of each user's activity space. | Tune `simulation.gravity_deterrence_exponent`, `profiles.work_distance_model`, `profiles.work_distance_exponential_lambda`, `profiles.work_distance_max_km`, and `simulation.rho`. These control commute span and exploratory range. |
| Dwell time | Time spent at stops. | Tune `activities.durations.*`, `activities.act_dur_scale`, `activities.act_dur_sigma_scale`, and `simulation.granularity_minutes`. Duration overrides directly reshape stay lengths. |
| Trip duration | Travel time, especially car-trip duration when available. | Tune `simulation.car_speed_kmh`, `simulation.walking_speed_kmh`, `simulation.bike_speed_kmh`, `road_network.enabled`, `rail_network.enabled`, and routing sidecar availability. Travel time follows path length and speed assumptions. |

## Activity and Motif Metrics

Lower Jensen-Shannon divergence is better: it means synthetic and observed
categorical probability mass is more similar.

| Metric | What it represents | Calibration levers |
| --- | --- | --- |
| Activity distribution | Overall share of visits by activity purpose/category. | Tune diary prompts in `diaries.city_profile_*`, `activities.kappa`, `activities.temperature`, and `activities.alignment_backend`. Fine-tune the activity aligner when semantically wrong activities are selected. |
| Activity transitions | Probability of moving from one activity category to another. | Tune schedule diary pools, `schedule.similarity_backend`, `schedule.temperature_beta_*`, and `schedule.alpha_beta_*`. Fine-tune the schedule aligner if profile-to-diary matching produces unrealistic sequences. |
| Daily activity profile | Activity mix by time of day. | Tune `simulation.granularity_minutes`, `activities.durations.*`, `activities.temperature`, and schedule aligner scores. Timing improves when macro diaries and micro durations agree with observed rhythms. |
| Daily motifs | Compact daily sequence patterns. | Tune `diaries.motif_exploration_rate`, `diaries.location_count_*`, `diaries.max_one_location_diaries`, and `schedule.alpha_beta_*`. Motifs improve when daily complexity and repeated routines match observed or literature patterns. |

## Spatial Metrics

For CPC, higher is better. For STVD and STVD-EMD, lower is better.

| Metric | What it represents | Calibration levers |
| --- | --- | --- |
| Common Part of Commuters | Overlap between synthetic and observed commute origin-destination flows. | Tune `profiles.home_building_weight`, `profiles.home_poi_inverse_weight`, `profiles.work_poi_weight`, `profiles.work_building_weight`, `profiles.work_distance_*`, and `profiles.work_from_home_probability`. Better home/work assignment directly improves OD overlap. |
| STVD distances / STVD-EMD | Spatial-temporal mismatch in visit volume over H3 cells and time. | Tune home/work placement, `simulation.gravity_deterrence_exponent`, `simulation.dt_update_mob_sim_hours`, schedule timing, and `comparison.evaluation_adaptation.*`. STVD improves when people are in the right cells at the right times. |
| Home locations | Residential density by H3 cell. | Tune `profiles.home_anchors_path`, `profiles.home_anchor_relevance`, `profiles.home_building_weight`, `profiles.home_poi_inverse_weight`, and `profiles.home_anchor_h3_resolution`. These determine where synthetic homes are assigned. |
| Work locations | Employment density by H3 cell. | Tune `profiles.work_poi_weight`, `profiles.work_building_weight`, `profiles.work_distance_model`, `profiles.work_distance_exponential_lambda`, and `profiles.work_distance_density_correction_power`. These determine employment anchors and commute length. |

## Mobility-Law Charts

Shape-match is the goal: fitted synthetic points should follow the observed
curve and literature reference where appropriate.

| Chart | What it represents | Calibration levers |
| --- | --- | --- |
| Travel-distance mobility law | Distribution tail of travel distances. | Tune `simulation.gravity_deterrence_exponent`, `simulation.rho`, `simulation.gamma`, and road/rail routing. These set short-vs-long trip frequency. |
| Radius-of-gyration mobility law | Distribution of user activity-space radii. | Tune `profiles.work_distance_*`, `simulation.gravity_deterrence_exponent`, and `simulation.rho`. These set the spread of home/work and exploratory locations. |
| Daily visited locations | Log-normal-like distribution of daily distinct locations. | Tune `diaries.location_count_mu`, `diaries.location_count_sigma`, `diaries.max_locations`, `diaries.max_one_location_diaries`, and schedule exploration. |
| Distance-frequency visitation law | Relationship between travel distance and visit frequency. | Tune `simulation.gamma`, `schedule.alpha_beta_*`, and `simulation.alpha`. Preferential return and social location choice affect repeated nearby visits. |

## Transport Metrics

Mode-share shape and mode-specific distance distributions should match observed
behavior.

| Metric | What it represents | Calibration levers |
| --- | --- | --- |
| Trips by transport mode / Share | Fraction and count of trips assigned to each mode. | Tune `profiles.car_probability`, `profiles.bike_probability`, vehicle ownership alignment, `simulation.walking_threshold_*`, `simulation.bike_threshold_*`, `road_network.enabled`, and `rail_network.enabled`. |
| Jump length by transport mode / Mean jump | Distance distribution within each transport mode. | Tune mode thresholds and routing availability. Walking and biking thresholds separate short trips from motorized trips. |
| Mean duration | Average travel time by transport mode. | Tune `simulation.car_speed_kmh`, `simulation.walking_speed_kmh`, `simulation.bike_speed_kmh`, road/rail graph quality, and `max_leg_waypoints`. |

## Time-Use Metrics

Lower absolute difference is better; signed charts should move toward zero.

| Metric | What it represents | Calibration levers |
| --- | --- | --- |
| Mean daily minutes | Synthetic and reference minutes per activity category. | Tune `activities.durations.*`, `activities.act_dur_scale`, diary prompts, and activity alignment. |
| Synthetic difference from time-use | Signed synthetic-minus-survey error by category. | Tune the categories with the largest absolute bars first using duration overrides or diary prompt edits. |
| Mean absolute time-use share difference | Average absolute percentage-point error across time-use categories. | Tune `activities.durations.*`, `activities.kappa`, `activities.temperature`, and fine-tune the activity aligner when selected activities are semantically wrong. |

## Social-Network Metrics

For synthetic-vs-observed Wasserstein, lower is better. Synthetic-vs-random
metrics are diagnostic: they indicate whether the graph is too random-like or
too structured relative to a degree-preserving baseline.

| Metric | What it represents | Calibration levers |
| --- | --- | --- |
| Degree | Number of ties per agent. | Tune `social.degree_mu_ln`, `social.degree_sigma_ln`, `social.max_degree`, `social.social_graph_k`, and `social.max_dynamic_degree`. |
| Clustering coefficient | How often an agent's friends are also connected to each other. | Tune `social.similarity_temperature`, `social.home_h3_resolution`, `social.work_h3_resolution`, `social.max_ring_expansion`, and dynamic friendship thresholds. |
| Edge persistence | Fraction of time windows in which a tie or co-presence edge recurs. | Tune `social.encounter_window_hours`, `social.regularity_threshold`, `social.friendship_update_interval_hours`, and schedule repeatability. |
| Topological overlap | Shared-neighbor overlap for connected agents. | Tune `social.topological_overlap_threshold`, profile similarity, and `social.recast_random_chance_probability`. |
| Observed network construction | Co-presence graph used for observed validation. | Tune `comparison.network_validation.location_mode`, `comparison.network_validation.location_col`, `comparison.network_validation.h3_resolution`, and `comparison.network_validation.max_group_size`. Bad grouping can make observed baselines too dense or too sparse. |

## Mobility-Profile Metrics

Shape-match is the goal across synthetic and observed profile groups.

| Metric | What it represents | Calibration levers |
| --- | --- | --- |
| Intermittency | Irregularity in a user's movement rhythm. | Tune `schedule.alpha_beta_*`, `simulation.gamma`, and diary diversity. More schedule reuse usually lowers intermittency. |
| Degree of return | Tendency to revisit the same locations. | Tune `simulation.gamma`, `schedule.alpha_beta_b`, and `diaries.location_count_*`. Stronger preferential return increases repeat visits. |
| Jump length | Movement distance within profile groups. | Tune gravity distance decay, routing, and transport thresholds. |
| Radius of gyration | Activity-space spread within profile groups. | Tune work-distance placement, gravity, and exploration. |
| Visits | Visit count within profile groups. | Tune daily location-count distribution, schedule exploration, and activity materialization. |

## Fine-Tuning Levers

Use config tuning first. Fine-tune when errors are systematic and semantic:

- Schedule aligner: improve profile-to-diary matching when daily sequences,
  motifs, or time-of-day activity profiles are wrong despite reasonable diary
  pools.
- Activity aligner: improve micro-activity selection when purpose is right but
  detailed activities or time-use categories are wrong.
- Profile coherence aligner: improve demographic consistency when generated
  profiles produce unrealistic home/work, vehicle, or activity behavior.
- Vehicle ownership aligner: improve transport mode shares when car/bike
  ownership is the bottleneck.
- Diary-generation prompts/data: improve macro-routines, location-count shape,
  weekend/weekday contrast, and special-day behavior when the candidate diary
  bank itself lacks the target pattern.
