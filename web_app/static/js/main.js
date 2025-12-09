function sendControl(action) {
    fetch(`/control/${action}`, {
        method: 'POST'
    })
        .then(response => response.json())
        .then(data => {
            console.log('Action triggered:', data);
        })
        .catch(error => {
            console.error('Error:', error);
        });
}
