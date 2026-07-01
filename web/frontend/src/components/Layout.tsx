import { Link, NavLink, Outlet } from "react-router-dom";

export function Layout() {
  return (
    <>
      <header className="top-nav">
        <div className="container">
          <Link to="/" className="brand">
            CityBehavEx
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
