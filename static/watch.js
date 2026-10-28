// the watchlist forms ask Yahoo for prices, which takes a moment: say so, and stop a second press going out
(function () {
  var forms = document.querySelectorAll('form[data-wait]');

  Array.prototype.forEach.call(forms, function (form) {
    form.addEventListener('submit', function () {
      form.querySelector('button[type=submit]').disabled = true;
      document.getElementById(form.getAttribute('data-wait')).hidden = false;
    });
  });

  window.addEventListener('pageshow', function () {       // the back button brings the page back as it was
    Array.prototype.forEach.call(forms, function (form) {
      form.querySelector('button[type=submit]').disabled = false;
      document.getElementById(form.getAttribute('data-wait')).hidden = true;
    });
  });
})();
