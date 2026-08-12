/* theme.js — the light/dark toggle, and nothing else.
 * Soft Bright is the default (no attribute); dark adds data-theme="dark" to
 * <html>. The choice is remembered across reloads in localStorage. */
window.StatXI = window.StatXI || {};

(function () {
  var KEY = 'statxi-theme';
  var root = document.documentElement;

  function apply(t){
    var btn = document.getElementById('themeToggle');
    if (t === 'dark'){ root.setAttribute('data-theme','dark'); if (btn) btn.textContent = '☀️ Bright'; }
    else             { root.removeAttribute('data-theme');     if (btn) btn.textContent = '🌙 Dark';  }
    localStorage.setItem(KEY, t);
  }

  StatXI.theme = {
    init: function(){
      apply(localStorage.getItem(KEY) || 'bright');   // default = Soft Bright
      var btn = document.getElementById('themeToggle');
      if (btn) btn.addEventListener('click', function(){
        apply(root.hasAttribute('data-theme') ? 'bright' : 'dark');
      });
    }
  };
})();
