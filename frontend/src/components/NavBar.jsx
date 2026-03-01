import { NavLink } from 'react-router-dom';

export default function NavBar() {
  return (
    <nav>
      <span className="nav-brand">NBA Sentiment</span>
      <div className="nav-links">
        <NavLink
          to="/"
          end
          className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
        >
          Overview
        </NavLink>
        <NavLink
          to="/series"
          className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
        >
          Series Explorer
        </NavLink>
        <NavLink
          to="/findings"
          className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
        >
          Findings
        </NavLink>
        <NavLink
          to="/speakers"
          className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
        >
          Speakers
        </NavLink>
      </div>
    </nav>
  );
}
