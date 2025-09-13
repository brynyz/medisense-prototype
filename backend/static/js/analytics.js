document.addEventListener('DOMContentLoaded', function() {
    // Common chart options for all charts
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false, // This is key - allows chart to fit container
        plugins: {
            legend: {
                display: false // Hide legend to save space
            }
        },
        scales: {
            x: {
                display: false // Hide x-axis
            },
            y: {
                display: false // Hide y-axis
            }
        },
        elements: {
            point: {
                radius: 2 // Smaller points
            },
            line: {
                borderWidth: 2 // Thinner lines
            }
        }
    };

    // Patient Trends Chart
    const patientCtx = document.getElementById('patientChart');
    if (patientCtx) {
        new Chart(patientCtx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    data: [1800, 1500, 1200, 1100, 1000, 1200],
                    borderColor: '#0a400c',
                    backgroundColor: 'rgba(10, 64, 12, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                ...commonOptions
            }
        });
    }

    // Most Common Symptoms Chart (Doughnut)
    const symptomsCtx = document.getElementById('symptomsChart');
    if (symptomsCtx) {
        new Chart(symptomsCtx, {
            type: 'doughnut',
            data: {
                labels: ['Fever', 'Cough', 'Headache'],
                datasets: [{
                    data: [40, 30, 30],
                    backgroundColor: [
                        '#0a400c',
                        '#549c78',
                        '#b1ab86'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                ...commonOptions,
                cutout: '60%' // Make doughnut hole bigger to fit better
            }
        });
    }

    // Inventory Status Chart (Bar)
    const inventoryCtx = document.getElementById('inventoryChart');
    if (inventoryCtx) {
        new Chart(inventoryCtx, {
            type: 'bar',
            data: {
                labels: ['In Stock', 'Low Stock', 'Out of Stock'],
                datasets: [{
                    data: [85, 12, 3],
                    backgroundColor: [
                        '#28a745',
                        '#ffc107',
                        '#dc3545'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                ...commonOptions
            }
        });
    }

    // Forecast Chart (Line with prediction)
    const forecastCtx = document.getElementById('forecastChart');
    if (forecastCtx) {
        new Chart(forecastCtx, {
            type: 'line',
            data: {
                labels: ['Jun', 'Jul', 'Aug', 'Sep'],
                datasets: [{
                    label: 'Actual',
                    data: [100, 110, null, null],
                    borderColor: '#0a400c',
                    backgroundColor: 'rgba(10, 64, 12, 0.1)',
                    fill: false,
                    tension: 0.4
                }, {
                    label: 'Forecast',
                    data: [null, 110, 132, 140],
                    borderColor: '#ffc107',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    borderDash: [5, 5], // Dashed line for forecast
                    fill: false,
                    tension: 0.4
                }]
            },
            options: {
                ...commonOptions
            }
        });
    }
});