'use strict';

function create_message(header, bodyText) {
    let div = document.createElement('div');
    div.className = "message";
    div.innerHTML = '<strong>' + header + '</strong> ' + bodyText;
    return div
}

function create_res_textline(header, bodyText) {
    let res = "";

    res += '<li>' + header + '<br>';
    res += '<p style="margin-left: 50px;">' + bodyText + '</p></li>';

    return res;
}

function create_single_experiment_res(response) {
    let tissue = response['tissue'];
    let chromosome = response['chromosome'];
    let position = response['position'];
    let orig_nuc = response['orig_nuc'];
    let variation = response['variation'];
    let features = response['most_features'];
    let pic_name_unique = response['pic_name_unique'];

    let most_features_str = "";
    for( let i = 0; i < features.length; i++ ) {
        most_features_str += features[i];
        if (i < features.length - 1) {most_features_str += ', '}
    }

    let res_div = document.createElement('div');
    res_div.className = "singleResult";
    res_div.innerHTML = '<h3>Результаты эксперимента:</h3>';
    res_div.innerHTML += '<p></p>';

    res_div.innerHTML += '<ul>';
    res_div.innerHTML +=
        create_res_textline('Использовалась модель для распознания ткани:', tissue);
    res_div.innerHTML +=
        create_res_textline('Хромосома:', chromosome);
    res_div.innerHTML +=
        create_res_textline('Позиция:', position);
    res_div.innerHTML +=
        create_res_textline('Референсный нуклеотид:', orig_nuc);
    res_div.innerHTML +=
        create_res_textline('Геномная вариация:', variation);
    res_div.innerHTML +=
        create_res_textline('<strong>Наиболее значимые позиции (Z-оценка > 3):</strong>', most_features_str);
    res_div.innerHTML += '</ul>';



    let anchor = document.getElementById("anchor");
    anchor.after(res_div)
}

function delete_all_generic_content() {
    let elems = document.querySelectorAll('.message');
    for( let i = 0; i < elems.length; i++ ) {
        elems[i].outerHTML = "";
    }
    elems = document.querySelectorAll('.singleResult');
    for( let i = 0; i < elems.length; i++ ) {
        elems[i].outerHTML = "";
    }
}

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
            console.log("Here");
            responseJson = await response.json();
            callback(responseJson);
        } else {
            throw new Error('Incorrect response');
        }
    } catch (exception) {
        console.log(exception.toString());

        delete_all_generic_content();
        let error_message = create_message("Ошибка сервера!", "Данные указаны неверно.");
        error_message.innerHTML = '<p></p>' + error_message.innerHTML;
        let anchor = document.getElementById("submit_btn1");
        anchor.after(error_message)
    }
}

function single_experiment(e) {
    e.preventDefault();
    delete_all_generic_content();

    let error_message = create_message("Расчёт...", "Пожалуйста, подождите.");
    error_message.innerHTML = '<p></p>' + error_message.innerHTML;
    let anchor = document.getElementById("submit_btn1");
    anchor.after(error_message);

    // формирование данных для запроса к серверу
    let formData = new FormData();
    let assembly = document.getElementById("select_assembly").value;
    let tissue = document.getElementById("select_tissue").value;
    let chromosome = document.getElementById("select_chromosome").value;
    let position = document.getElementById("position").value;
    let nucleotide = document.getElementById("select_nucleotide").value;

    // валидация
    if (position.empty || isNaN(parseInt(position)) || position.indexOf(".") !== -1 ||
        position.indexOf(",") !== -1 || position.indexOf("e") !== -1 || parseInt(position) < 1) {

        delete_all_generic_content();

        let error_message = create_message("Ошибка!", "Позиция введена неверно.");
        error_message.innerHTML = '<p></p>' + error_message.innerHTML;
        let anchor = document.getElementById("submit_btn1");
        anchor.after(error_message);
    }
    // отправка запроса на сервер
    else {
        position = String(parseInt(position));
        formData.append('assembly', assembly);
        formData.append('tissue', tissue);
        formData.append('chromosome', chromosome);
        formData.append('position', position);
        formData.append('nucleotide', nucleotide);
        console.log(position);

        ajaxForm("http://localhost:8000/api/send-self-input-form", "POST", formData, (response) => {
            console.log(response);

            delete_all_generic_content();

            create_single_experiment_res(response);
        });
    }
}

let submit_btn1 = document.getElementById("submit_btn1");
submit_btn1.onclick = single_experiment;
