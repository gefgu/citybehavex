import { Link, NavLink, Outlet } from "react-router-dom";

const assetBase = import.meta.env.BASE_URL.replace(/\/?$/, "/");

export function Layout() {
  return (
    <>
      <header className="top-nav">
        <div className="container">
          <Link to="/" className="brand">
            <img src={`${assetBase}citybx_logo.png`} alt="CityBehavEx" className="brand-logo" />
          </Link>
          <nav>
            <NavLink to="/" end>
              Home
            </NavLink>
            <NavLink to="/experiments">Experiments</NavLink>
          </nav>
        </div>
      </header>
      <main className="container">
        <Outlet />
      </main>
    </>
  );
}
