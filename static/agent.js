// the assistant page: sends the chat to /api/agent and shows the answer and the steps it took.
// everything is built from text nodes, so nothing the model says is ever treated as html.
(function () {
  var log = document.getElementById('log');
  var form = document.getElementById('ask');
  var box = document.getElementById('q');
  var send = document.getElementById('send');
  var hello = document.getElementById('hello');
  var history = [];        // what gets sent every time: plain {role, content} turns
  var busy = false;

  function el(tag, cls, text) {
    var x = document.createElement(tag);
    if (cls) x.className = cls;
    if (text !== undefined) x.textContent = text;
    return x;
  }

  // **bold** inside a line
  function inline(node, text) {
    text.split(/(\*\*[^*]+\*\*)/).forEach(function (part) {
      if (/^\*\*[^*]+\*\*$/.test(part)) node.appendChild(el('strong', '', part.slice(2, -2)));
      else if (part) node.appendChild(document.createTextNode(part));
    });
    return node;
  }

  // paragraphs and "- " bullets, which is all the assistant is asked to write
  function render(into, text) {
    var list = null;
    text.split('\n').forEach(function (line) {
      var m = line.match(/^\s*[-*]\s+(.*)$/);
      if (m) {
        if (!list) {
          list = el('ul');
          into.appendChild(list);
        }
        list.appendChild(inline(el('li'), m[1]));
      } else {
        list = null;
        if (line.trim()) into.appendChild(inline(el('p'), line));
      }
    });
  }

  function bottom() { log.scrollTop = log.scrollHeight; }

  // a long answer should be read from its first line, not its last
  function showStart(m) {
    log.scrollTop += m.getBoundingClientRect().top - log.getBoundingClientRect().top - 8;
  }

  function bubble(who, text) {
    var m = el('div', 'msg ' + who);
    var body = el('div', 'body');
    if (who === 'bot') render(body, text);
    else body.textContent = text;
    m.appendChild(body);
    log.appendChild(m);
    bottom();
    return m;
  }

  function shown(label, value) {
    var d = el('div', 'kv');
    d.appendChild(el('span', 'k', label));
    d.appendChild(el('pre', '', typeof value === 'string' ? value : JSON.stringify(value, null, 2)));
    return d;
  }

  // what it did, and what it cost
  function details(m, data) {
    if (data.steps.length) {
      var d = el('details', 'steps');
      d.appendChild(el('summary', '', 'Steps taken (' + data.steps.length + ')'));
      var ol = el('ol');
      data.steps.forEach(function (s) {
        var li = el('li');
        var head = el('div', 'shead');
        head.appendChild(el('b', '', s.tool));
        head.appendChild(el('span', s.error ? 'bad' : 'ok', s.error ? '✗ error' : '✓ ok'));   // a mark and a word, not only a colour
        head.appendChild(el('span', 'ms', s.ms + ' ms'));
        li.appendChild(head);
        li.appendChild(shown('Input', s.input));
        li.appendChild(shown(s.error ? 'Error' : 'Result', s.output));
        ol.appendChild(li);
      });
      d.appendChild(ol);
      m.appendChild(d);
    }
    var u = data.usage;
    m.appendChild(el('div', 'meta', u.calls + ' model call' + (u.calls === 1 ? '' : 's') + ' · ' +
                     u.input_tokens.toLocaleString() + ' tokens in, ' + u.output_tokens.toLocaleString() + ' out'));
  }

  function ask(q) {
    if (!q || busy) return;
    busy = true;
    send.disabled = true;
    box.value = '';
    if (hello) {
      hello.remove();
      hello = null;
    }
    history.push({role: 'user', content: q});
    bubble('user', q);
    var wait = bubble('bot wait', 'Working on it');
    wait.setAttribute('role', 'status');

    fetch('/api/agent', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({messages: history})
    }).then(function (r) {
      return r.json().catch(function () {
        return {error: 'The request failed (HTTP ' + r.status + ').'};
      }).then(function (data) { return {ok: r.ok, data: data}; });
    }).then(function (res) {
      wait.remove();
      if (!res.ok) throw new Error(res.data.error || 'Something went wrong.');
      history.push({role: 'assistant', content: res.data.reply});
      var m = bubble('bot', res.data.reply);
      details(m, res.data);
      showStart(m);
    }).catch(function (err) {
      wait.remove();
      history.pop();           // that question never got an answer, so take it back
      box.value = q;           // and put it back in the box to try again
      bubble('err', err instanceof TypeError ? "Couldn't reach the app. Is it still running?" : err.message);
    }).then(function () {
      busy = false;
      send.disabled = box.disabled;
      box.focus();
    });
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    ask(box.value.trim());
  });

  box.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  document.querySelectorAll('.chip').forEach(function (c) {
    c.addEventListener('click', function () {
      box.value = c.getAttribute('data-q');
      box.focus();
    });
  });

  var helloNode = hello;
  document.getElementById('clear').addEventListener('click', function () {
    if (busy) return;
    history = [];
    log.textContent = '';
    hello = helloNode;
    if (hello) log.appendChild(hello);
    box.value = '';
    box.focus();
  });
})();
