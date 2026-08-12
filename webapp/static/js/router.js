/* router.js — the "single page" part of single-page app.
 *
 * Instead of loading a new HTML file per page, we swap the contents of
 * <main id="view"> based on the URL hash (#/  and  #/predict). Hash routing is
 * used (not the History API) because it needs no server config and works over
 * file:// — the browser never asks the server for "/predict", it just fires a
 * `hashchange` event we listen for.
 *
 * A "view" is any object with { render(): htmlString, mount?(rootEl): void }.
 * render() returns markup; the optional mount() wires up event listeners after
 * that markup is on the page. */
window.StatXI = window.StatXI || {};

(function () {
  var routes = {};

  // the path part of the hash, WITHOUT any ?query (so #/detail?home=X matches /detail)
  function currentPath(){ return location.hash.replace(/^#/, '').split('?')[0] || '/'; }

  function render(){
    var view = routes[currentPath()] || routes['/'];   // unknown hash -> home
    var root = document.getElementById('view');
    root.innerHTML = view.render();
    if (view.mount) view.mount(root);

    // highlight the active nav link
    Array.prototype.forEach.call(document.querySelectorAll('.nav a'), function(a){
      a.classList.toggle('active', a.getAttribute('href') === '#' + currentPath());
    });
    window.scrollTo(0, 0);
  }

  StatXI.router = {
    add:   function(path, view){ routes[path] = view; return this; },  // chainable
    start: function(){ window.addEventListener('hashchange', render); render(); },
    // parse the "?home=X&away=Y" part of the current hash into an object
    query: function(){
      var out = {}, q = (location.hash.split('?')[1] || '');
      q.split('&').forEach(function(p){
        if (!p) return;
        var kv = p.split('='); out[decodeURIComponent(kv[0])] = decodeURIComponent(kv[1] || '');
      });
      return out;
    }
  };
})();
