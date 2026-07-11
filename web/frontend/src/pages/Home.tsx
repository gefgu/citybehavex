import { Link } from "react-router-dom";

const assetBase = import.meta.env.BASE_URL.replace(/\/?$/, "/");

export function Home() {
  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <h1>Scalable urban simulation, validated against the real thing.</h1>
          <p className="lead">
            CityBehavEx is an LLM-assisted urban simulation platform that scales to
            city-size populations while keeping behavior inspectable. It combines
            mobility models, semantic alignment, trajectory replay and empirical
            validation against spatial, temporal, activity and social patterns.
          </p>
          <div className="hero-actions">
            <Link to="/experiments" className="btn btn-primary">
              Browse experiments
            </Link>
            <a href="#" className="btn btn-secondary">
              Read paper
            </a>
          </div>
        </div>
        <img
          className="hero-figure"
          src={`${assetBase}map_timeline_view.jpg`}
          alt="CityBehavEx time-line map view"
        />
      </section>

      <section className="signature-grid">
        <div className="signature-card sig-coral">
          <h3>Distributions &amp; laws</h3>
          <p>
            Spatial and temporal validation with travel distance, radius of gyration,
            trip duration, dwell time, visitation frequency and mobility laws.
          </p>
        </div>
        <div className="signature-card sig-forest">
          <h3>Activity &amp; profiles</h3>
          <p>
            Profile-aware diaries, MTUS-grounded micro-schedules, activity transitions,
            daily motifs and Routiner / Regular / Scouter mobility profiles.
          </p>
        </div>
        <div className="signature-card sig-dark">
          <h3>Spatial difference</h3>
          <p>
            Interactive replay, H3 spatial differences, HOME and WORK locations, transport choices and social
            network metrics for debugging synthetic behavior.
          </p>
        </div>
      </section>
    </>
  );
}
