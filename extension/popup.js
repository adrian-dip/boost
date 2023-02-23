(function () {
    chrome.tabs.query({active: true, lastFocusedWindow: true}, tabs => {
    let url = tabs[0].url;
    let { hostname } = new URL(url);
    document.getElementById("website").innerHTML = hostname
    gHostname = hostname

    chrome.storage.local.get(['enabled', 'deactivateUrls']
    ).then(function (r) {
        if (typeof r.enabled === 'undefined') {
            chrome.storage.local.set({enabled: true})
            document.getElementById("masterSwitch").checked = true
        }
        if (r.enabled === false) {
            document.getElementById("masterSwitch").checked = false;
        }
        else {
            document.getElementById("masterSwitch").checked = true;
        }

        if (typeof r.deactivateUrls === "undefined") {
            var arraydeactivateUrls = {0:0}
            chrome.storage.local.set({deactivateUrls: arraydeactivateUrls})
        }
        
        else {
            var urls = r.deactivateUrls
            if (hostname in urls) {
                document.getElementById('urlSwitch').checked = true
            }
            else {
                document.getElementById('urlSwitch').checked = false
            }
        }
})})})()


var enabled = true; //enabled by default
var masterSwitch = document.getElementById('masterSwitch')
var urlSwitch = document.getElementById('urlSwitch')


chrome.storage.local.get('enabled', data => {
    enabled = !!data.enabled
})

urlSwitch.onchange = () => {
            chrome.storage.local.get('deactivateUrls', data => {
                var urls = data.deactivateUrls;
                if (gHostname in urls) {
                    delete urls[gHostname] 
                    chrome.storage.local.set({deactivateUrls: urls})
                }

                else {
                urls[gHostname] = 0;
                chrome.storage.local.set({deactivateUrls: urls})
            }})
}

masterSwitch.onchange = () => {
    enabled = !enabled
    chrome.storage.local.set({enabled:enabled})
}