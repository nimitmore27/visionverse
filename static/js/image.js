
const input = document.getElementById('upload');
const infoArea = document.getElementById('upload-label');
const resultDiv = document.getElementById('resultDiv');
const mainAlert = document.getElementById('mainAlert');
const imageResult = document.getElementById('imageResult');
input.addEventListener("change", function (event) {
    if (input.files[0].type != "image/jpeg") {
        alert("Please upload a jpeg image");
        return;
    }
    readURL(input);
    showFileName(event);
    resultDiv.innerHTML = "";
    mainAlert.classList.remove("alert-primary");
    mainAlert.classList.add("alert-warning");
    mainAlert.innerText = "Please wait...";
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("sample_image", input.files[0]);
    xhr.open("POST", "/analyzeImage");
    xhr.send(formData);
    xhr.addEventListener("readystatechange", function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
            let jres = JSON.parse(xhr.responseText);
            jres = jres['result']
            console.log(jres)
            jres.forEach(function (element) {
                const div = document.createElement("div");
                div.className = "alert alert-primary text-center";
                div.setAttribute("role", "alert");
                div.innerHTML = "Prediction (index, name): " + element[0] + ", Score: " + element[1];
                resultDiv.appendChild(div);
            });
            mainAlert.classList.remove("alert-warning");
            mainAlert.classList.add("alert-success");
            mainAlert.innerText = "Here is the analysis result!";
        }
    });
});
function showFileName(event) {
    var input = event.srcElement;
    var fileName = input.files[0].name;
    console.log(input.files[0].type);
    infoArea.textContent = 'File name: ' + fileName;
}
function readURL(input) {
    if (input.files[0].type != "image/jpeg") {
        alert("Please upload a jpeg image");
        return;
    }
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            imageResult.src = e.target.result;
        };
        reader.readAsDataURL(input.files[0]);
    }
}