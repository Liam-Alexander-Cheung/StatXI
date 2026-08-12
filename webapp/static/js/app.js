/* app.js — the bootstrap. Runs last, once every other file has defined its
 * piece of StatXI. Turns the theme on, registers the routes, and starts the
 * router (which renders whatever the current URL hash points at). */
(function () {
  StatXI.theme.init();

  StatXI.router
    .add('/',        StatXI.views.landing)
    .add('/predict', StatXI.views.predict)
    .add('/detail',  StatXI.views.detail)
    .start();
})();
