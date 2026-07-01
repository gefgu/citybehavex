import { Link } from "react-router-dom";

export function Home() {
  return (
    <>
      <section className="hero">
        <h1>Synthetic urban mobility, measured against the real thing.</h1>
        <p className="lead">
          CityBehavEx simulates agent trajectories for a city and compares them to
          observed mobility data. This is the interactive report — distributions,
          mobility laws, activity patterns, profiles and spatial differences, straight
          from the simulation outputs.
        </p>
        <Link to="/experiments" className="btn btn-primary">
          Browse experiments
        </Link>
      </section>

      <section className="signature-grid">
        <div className="signature-card sig-coral">
          <h3>Distributions &amp; laws</h3>
          <p>
            ECDFs and fitted mobility laws — jump lengths, radius of gyration, trip
            and dwell time, visitation frequency.
          </p>
        </div>
        <div className="signature-card sig-forest">
          <h3>Activity &amp; profiles</h3>
          <p>
            Visit-purpose mixes, activity transitions, daily rhythms, motifs and
            Routiner / Regular / Scouter mobility profiles.
          </p>
        </div>
        <div className="signature-card sig-dark">
          <h3>Spatial difference</h3>
          <p>
            An H3 choropleth of where and when synthetic volume and peak timing
            diverge from the observed city.
          </p>
        </div>
      </section>
    </>
  );
}
