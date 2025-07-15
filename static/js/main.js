function updateClock() {
  const clockElement = document.getElementById('clock');
  if (clockElement) {
    const now = new Date();
    const timeString = now.toLocaleTimeString(); // e.g., 10:45:21 PM
    clockElement.textContent = timeString;
  }
}

// Wait for DOM to be fully loaded before starting the clock
document.addEventListener('DOMContentLoaded', function() {
  updateClock(); // Run immediately on load
  setInterval(updateClock, 1000); // Update every 1 second
});