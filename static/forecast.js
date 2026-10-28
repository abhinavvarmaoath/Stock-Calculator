// the forecast pages: a "this takes a while" message on the form, and two charts on the results.
// plain svg, and the numbers come from the json block in the page.
(function () {
  var ns = 'http://www.w3.org/2000/svg';

  function mk(tag, attrs, parent) {
    var el = document.createElementNS(ns, tag);
    for (var k in attrs) el.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(el);
    return el;
  }

  // axis ticks on round numbers
  function ticks(lo, hi, n) {
    var raw = (hi - lo) / n;
    if (!(raw > 0)) return {at: [lo], step: 1};
    var p = Math.pow(10, Math.floor(Math.log10(raw)));
    var step = p * [1, 2, 2.5, 5, 10].filter(function (m) { return m * p >= raw; })[0];
    var at = [];
    for (var i = Math.ceil(lo / step - 1e-9); i * step <= hi + step * 1e-9; i++) at.push(i * step);
    return {at: at, step: step};
  }

  // draw once, and again whenever the width changes
  function watch(plot, draw) {
    var lastw = 0;
    function go() {
      if (plot.clientWidth === lastw) return;
      lastw = plot.clientWidth;
      var old = plot.querySelector('svg');
      if (old) old.remove();
      draw();
    }
    new ResizeObserver(go).observe(plot);
    go();
  }

  function text(parent, cls, x, y, anchor, str) {
    var t = mk('text', {'class': cls, x: x, y: y, 'text-anchor': anchor}, parent);
    t.textContent = str;
    return t;
  }

  function nice(v) { return String(Math.round(v * 1000) / 1000); }

  function pct(v, dec) { return (v > 0 ? '+' : v < 0 ? '-' : '') + Math.abs(v).toFixed(dec) + '%'; }

  // how much closer than "price stays put" each method was. bars to the right are better, to the
  // left worse, and the grey zone behind each is the gap luck alone could have produced
  function skill(fig, d) {
    var plot = fig.querySelector('.plot'), rows = d.rows;

    watch(plot, function () {
      var w = Math.max(plot.clientWidth, 280), rh = 56;
      var m = {l: 14, r: 14, t: 6, b: 50};
      var h = m.t + rows.length * rh + m.b, iw = w - m.l - m.r;
      // wide enough for every bar and for most of every grey zone, but not for the huge ones
      var ext = Math.max.apply(null, rows.map(function (r) { return Math.max(Math.abs(r.skill), Math.min(r.needs || 0, 60)); })) * 1.15 || 10;
      var xt = ticks(-ext, ext, Math.max(3, Math.floor(iw / 80)));
      ext = Math.max(ext, xt.at[xt.at.length - 1]);
      function X(v) { return m.l + iw * (v + ext) / (2 * ext); }
      var base = m.t + rows.length * rh;

      var svg = mk('svg', {width: w, height: h, viewBox: '0 0 ' + w + ' ' + h, 'aria-hidden': 'true'});
      plot.insertBefore(svg, plot.firstChild);

      xt.at.forEach(function (v) {
        mk('line', {'class': 'grid', x1: X(v), x2: X(v), y1: m.t, y2: base}, svg);
        text(svg, 'tick', X(v), base + 16, 'middle', (v > 0 ? '+' : '') + nice(v) + '%');
      });
      mk('line', {'class': 'zero', x1: X(0), x2: X(0), y1: m.t, y2: base}, svg);
      text(svg, 'atitle', m.l, h - 6, 'start', '← worse');
      text(svg, 'atitle', w - m.r, h - 6, 'end', 'better →');
      text(svg, 'atitle', X(0), h - 6, 'middle', 'same as benchmark');

      rows.forEach(function (r, i) {
        var top = m.t + i * rh;
        text(svg, 'rowlabel', m.l, top + 16, 'start', r.label);
        if (r.needs) {
          var z0 = X(Math.max(-r.needs, -ext)), z1 = X(Math.min(r.needs, ext));      // cut off at the edges of the chart
          mk('rect', {'class': 'luckband', x: z0, y: top + 22, width: z1 - z0, height: 26, rx: 3}, svg);
          text(svg, 'zone', w - m.r, top + 16, 'end', r.needs >= 100 ? 'no gap would count' : 'luck zone \u00b1' + r.needs.toFixed(0) + '%');
        }
        var x0 = X(0), x1 = X(r.skill);
        var kind = r.edge === 'better' || r.edge === 'worse' ? r.edge : 'tied';
        mk('rect', {'class': 'fbar ' + kind, x: Math.min(x0, x1), y: top + 26, width: Math.max(Math.abs(x1 - x0), 1), height: 18, rx: 2}, svg);
        // the value goes just past the end of the bar, or on the other side if there is no room
        var right = r.skill >= 0, label = pct(r.skill, 1), room = right ? w - m.r - x1 : x1 - m.l;
        if (room < 54) right = !right;
        text(svg, 'vals', right ? Math.max(x0, x1) + 6 : Math.min(x0, x1) - 6, top + 39, right ? 'start' : 'end', label);
      });
    });
  }

  // how often the real price landed inside the simulator's range, against how often it should have
  function range(fig, d) {
    var plot = fig.querySelector('.plot'), rows = d.calibration;

    watch(plot, function () {
      var w = Math.max(plot.clientWidth, 280), rh = 72;
      var m = {l: 14, r: 14, t: 6, b: 46};
      var h = m.t + rows.length * rh + m.b, iw = w - m.l - m.r;
      var base = m.t + rows.length * rh;
      function X(v) { return m.l + iw * v / 100; }

      var svg = mk('svg', {width: w, height: h, viewBox: '0 0 ' + w + ' ' + h, 'aria-hidden': 'true'});
      plot.insertBefore(svg, plot.firstChild);

      [0, 25, 50, 75, 100].forEach(function (v) {
        mk('line', {'class': 'grid', x1: X(v), x2: X(v), y1: m.t, y2: base}, svg);
        text(svg, 'tick', X(v), base + 16, 'middle', v + '%');
      });
      text(svg, 'atitle', m.l + iw / 2, h - 6, 'middle', 'How often the real price landed inside the range');

      rows.forEach(function (c, i) {
        var top = m.t + i * rh, mid = top + 36;
        text(svg, 'rowlabel', m.l, top + 14, 'start', c.level + '% range');
        mk('line', {'class': 'track', x1: X(0), x2: X(100), y1: mid, y2: mid}, svg);
        mk('rect', {'class': 'luckband', x: X(c.lo), y: mid - 12, width: Math.max(X(c.hi) - X(c.lo), 2), height: 24, rx: 3}, svg);
        mk('line', {'class': 'claim', x1: X(c.level), x2: X(c.level), y1: mid - 16, y2: mid + 16}, svg);
        mk('circle', {'class': 'obs' + (c.verdict === 'ok' ? '' : ' off'), cx: X(c.covered), cy: mid, r: 7}, svg);
        // the number sits under the dot, kept inside the chart
        var lx = Math.min(Math.max(X(c.covered), m.l + 20), w - m.r - 20);
        text(svg, 'vals', lx, mid + 32, 'middle', c.covered.toFixed(0) + '%');
      });
    });
  }

  // the form: tell people to wait, because the test takes a few seconds
  var form = document.getElementById('runform');
  if (form) {
    var go = document.getElementById('go'), msg = document.getElementById('running');
    form.addEventListener('submit', function () {
      go.disabled = true;
      msg.hidden = false;
    });
    window.addEventListener('pageshow', function () {       // back button brings the page back as it was
      go.disabled = false;
      msg.hidden = true;
    });
  }

  var data = document.getElementById('boarddata');
  if (!data) return;
  var d = JSON.parse(data.textContent);
  skill(document.getElementById('skill'), d);
  range(document.getElementById('range'), d);
})();
