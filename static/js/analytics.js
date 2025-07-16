document.addEventListener('DOMContentLoaded', function() {
    // Patient Trends Chart (Line Chart)
    const patientCtx = document.getElementById('patientChart').getContext('2d');
    new Chart(patientCtx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
            datasets: [{
                // Replace the array below with actual patient trend data or inject it server-side
                data: [10, 20, 30, 40, 50, 60, 70],
                borderColor: '#4CAF50',
                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { display: false }
            },
            elements: {
                point: { radius: 0 }
            }
        }
    });

    // Symptoms Chart (Bar Chart)
    const symptomsCtx = document.getElementById('symptomsChart').getContext('2d');
    new Chart(symptomsCtx, {
        type: 'bar',
        data: {
            labels: ['Fever', 'Cough', 'Headache'],
            datasets: [{
                data: [45, 32, 28],
                backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { display: false }
            }
        }
    });

    // Inventory Chart (Doughnut Chart)
    const inventoryCtx = document.getElementById('inventoryChart').getContext('2d');
    new Chart(inventoryCtx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [75, 25],
                backgroundColor: ['#4CAF50', '#f44336'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            cutout: '70%'
        }
    });

    // Forecast Chart (Area Chart)
    const forecastCtx = document.getElementById('forecastChart').getContext('2d');
    new Chart(forecastCtx, {
        type: 'line',
        data: {
            labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
            datasets: [{
                data: [30, 45, 55, 75],
                borderColor: '#ff6b6b',
                backgroundColor: 'rgba(255, 107, 107, 0.2)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { display: false }
            },
            elements: {
                point: { radius: 0 }
            }
        }
    });
});