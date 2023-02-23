(function () {
    chrome.storage.local.get(['enabled', 'deactivateUrls'], data => {
        if (data.enabled) {
            var urlcheck = data.deactivateUrls;
            const urlval = !(document.location.origin.split('/')[2] in urlcheck)
            if (urlval) {
                ansdict = {}

                ///////      Messagging     ///////

                function handleResponse(ans) {
                    n_filtered = 0
                    for (const res of Object.keys(ans)) {
                        if (ans['score'] === 0) {
                            ans['score'] = ansdict['score']
                            rocket_response_img.src = whitewarn
                            rocket_response_txt = negresponse

                        } if (ans['score'] === 1) {
                            ans['score'] = ansdict['score']
                            rocket_response_img.src = whitecheck
                            rocket_response_txt = neutralresponse

                        }
                          if (ans['score'] === 2) {
                            ans['score'] = ansdict['score']
                            rocket_response_img.src = whiteheart
                            rocket_response_txt = posresponse

                        }
                    }
                    console.log(message.payload)
                    return message.payload;
                }

                function handleError(error) {
                    console.log(`Error: ${error}`)
                }

                function notifyBackgroundPage(mess) {
                    const sending = chrome.runtime.sendMessage(mess)
                    sending.then(handleResponse, handleError)
                }

                function clearMessage() {
                    clearTimeout(timeoutID)
                }


                ///// Inbrowser popup /////
                fetch(chrome.runtime.getURL('/window.html')).then(r => r.text()).then(html => {
                    document.body.insertAdjacentHTML('beforeend', html)
                })

                const whitex = chrome.runtime.getURL("images/x.png")
                const whiterocket = chrome.runtime.getURL("images/whiterocket.png")
                const whiteheart = chrome.runtime.getURL("images/heart.png")
                const whitewarn = chrome.runtime.getURL("images/warning.png")
                const whitecheck = chrome.runtime.getURL("images/checkmark.png")
                const inver_r = chrome.runtime.getURL("images/inverse.png")
                const loadWhite = chrome.runtime.getURL("images/loading.png")
                const posresponse = "Positive response"
                const neutralresponse = "Neutral response"
                const negresponse = "Negative response"


                ///// Inbrowser popup elements /////
                const popupelmodalheader = document.createElement('div')
                popupelmodalheader.id = 'boostrocketic_header'



                /////// Button ///////
                fetch(chrome.runtime.getURL('/rocket_bt.html')).then(r => r.text()).then(html => {
                    document.body.insertAdjacentHTML('beforeend', html)

                    rocketlinkim = document.getElementById('boostrocketim_span0') // link tag
                    rocket_im = document.getElementById('rocket_im0') // img link
                    rocket_ic = document.getElementById('boostrocketic_span0') // parent div

                    rocket_im.src = inver_r
                    rocket_im.style.height = '30px'
                    rocket_im.style.borderRadius = '20%'
                    

                    rocketlinkim.addEventListener('click', function () {
                        var modal_hmurlrct0 = document.getElementById('window_rin')
                        modal_hmurlrct0.style.display = "block"
                        var coordinatemap = rocket_ic.getBoundingClientRect()
                        modal_hmurlrct0.style.top = (coordinatemap['top'] - 320).toString() + 'px'
                        modal_hmurlrct0.style.left = coordinatemap['left'].toString() + 'px'
                        modal_hmurlrct0.style.visibility = 'visible'
                        modal_hmurlrct0.style.opacity = '1'
                        rocket_ic.style.display = 'none'
                        rocket_ic.style.opacity = '0'
                        modal_hmurlrct0.style.zIndex = '99'
                        var modal_hmurlrct0l = document.getElementById('whiterocketimgl00')
                        var modal_hmurlrct0x = document.getElementById('whiterocketx00')
                        modal_hmurlrct0l.src = whiterocket
                        modal_hmurlrct0x.src = whitex
                        modal_hmurlrct0x.addEventListener('click', function () {
                            modal_hmurlrct0.style.display = "none"
                            modal_hmurlrct0.style.opacity = '0'
                            rocket_ic.style.display = "block"
                            rocket_ic.style.opacity = '0.8'
                            document.addEventListener('scroll', function () {
                                    modal_hmurlrct0.style.display = "none"
                                    modal_hmurlrct0.style.visibility = "hidden" })

                        })
                    })
                })


                let timeoutID = null;
                window.addEventListener('load', function () {
                    document.addEventListener("click", function (e) {

                        const searchQueries = ['input', '[contenteditable=true]', 'textarea']
                        const editableElements = {}

                        searchQueries.forEach(function (item, idx) {

                                editableElements[item] = document.querySelectorAll(item);
                                editableElements[item].forEach(el => {

                                        el.addEventListener('input', function (ev) {
                                            clearTimeout(timeoutID)
                                            timeoutID = setTimeout(function () {
                                                var { value } = ev.target;
                                                if (typeof value === 'string') {
                                                    if (value.split(' ').length > 6) {
                                                        console.log(ev);
                                                        rocket_response_img = document.getElementById('rocket_response_img')
                                                        rocket_response_txt = document.getElementById('rocket_response_txt')
                                                        rocket_response_img.src = loadWhite
                                                        var coordinates = ev.target.getBoundingClientRect()
                                                        var top_coordinate = coordinates['bottom'] - 10;
                                                        rocket_ic.style.top = top_coordinate.toString() + 'px'
                                                        var left_coordinate = coordinates['right'] - 10;
                                                        rocket_ic.style.left = left_coordinate.toString() + 'px'
                                                        rocket_ic.style.display = 'block'
                                                        rocket_ic.style.visibility = 'visible';
                                                        rocket_ic.style.opacity = '0.8';
                                                        document.addEventListener('scroll', function () {
                                                                rocket_ic.style.display = "none"
                                                                rocket_ic.style.visibility = "hidden" 

                                                        },
                                                        {
                                                            passive: true
                                                        })
                                                    }
                                                } console.log(value)}, 750)
})})})})})}}})})()