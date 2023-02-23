function getScore(message, response) {
    const options = {
        method: 'POST',
        headers: {
            "Content-Type": "application/json"
        },
        sentences: message.sentence,
        order: 'Boost'
    }

    fetch('http://dipadrian.pythonanywhere.com', 
    options).then(function(r) {
        return r.json()
    }).then(function (data) {
        var score = data;
        response.payload = score
        return response.payload 
    })}

    
chrome.runtime.onMessage.addListener(
    function(message, sender, sendResponse) {
        if (message.order === 'startup') {
            let response = {
                message: "response",
                payload: message.payload
            }
            getScore(message, response)
            sendResponse(response)
    }})