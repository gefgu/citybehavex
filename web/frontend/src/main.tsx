import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, createHashRouter, Navigate, RouterProvider } from "react-router-dom";
import "@picocss/pico/css/pico.min.css";
import "./styles/tokens.css";
import "./styles/app.css";
import { Layout } from "./components/Layout";
import { Home } from "./pages/Home";
import { Experiments } from "./pages/Experiments";
import { Charts } from "./pages/Charts";
import { Timeline } from "./pages/Timeline";

const routes = [
  {
    element: <Layout />,
    children: [
      { path: "/", element: <Home /> },
      { path: "/experiments", element: <Experiments /> },
      { path: "/experiments/:id/charts", element: <Charts /> },
      { path: "/experiments/:id/timeline", element: <Timeline /> },
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
];

const router =
  import.meta.env.VITE_STATIC_DEMO === "true"
    ? createHashRouter(routes)
    : createBrowserRouter(routes, { basename: import.meta.env.BASE_URL });

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
