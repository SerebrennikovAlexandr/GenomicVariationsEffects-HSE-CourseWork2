'use strict';

/**
 * Ajax с телом FormData
 * @param {string} route - адресс
 * @param {string} method - метод запроса
 * @param {FormData} formData - данные
 * @param {function} callback - функция, которая будет вызвана после запроса
 */
async function ajaxForm(route, method, formData, callback) {

    const reqBody = {
        method: method,
        mode: 'cors',
        credentials: 'include',
    };

    if (method !== 'GET' && method !== 'HEAD') {
        reqBody['body'] = formData;
    }

    const req = new Request(route, reqBody);
    let responseJson = null;
    try {
        const response = await fetch(req);
        if (response.ok) {
            responseJson = await response.json();
            callback(responseJson);
        } else {
            throw new Error('Response not ok');
        }
    } catch (exception) {
        let test = document.querySelectorAll('.textProgress');
        for( let i = 0; i < test.length; i++ )
            { test[i].outerHTML = ""; }
    }
}

function single_experiment(e) {
    e.preventDefault();

    var formData = new FormData();

    ajaxForm("http://localhost:8000/api/send-self-input-form", "POST", formData, (response) => {

    });

    console.log("kek")
}

let submit_btn1 = document.getElementById("submit_btn1");
submit_btn1.onclick = single_experiment;
