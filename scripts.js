function handleFormSubmit(event, loaderId) {
  event.preventDefault(); // Stops the form from submitting normally
  document.getElementById(loaderId).style.display = 'block';
  // Add AJAX request or form submission logic here
}

function myFunction() {
  var x = document.getElementById("navDemo");
  if (x.className.indexOf("w3-show") === -1) {
      x.className += " w3-show";
  } else { 
      x.className = x.className.replace(" w3-show", "");
  }
}