function toDataPage() {
    window.location.href="/evaluate"
}

function uploadDataset(){
    alert("To Be Developed")
}

function uploadModels() {
    alert("To Be Developed")
}

function downloadModels() {
    alert("To Be Developed")
}

function pauseContinue(url) {
    var pauseEle = $("#pause");
    // if (pauseEle.attr("value") === "暂停任务"){
    //     pauseEle.val("继续任务")
    //     $("#download_model").show()
    // }else{
    //     pauseEle.val("暂停任务")
    //     $("#download_model").hide()
    // }
    $.get(url)
    if (pauseEle.attr("value") === "暂停任务"){
        pauseEle.val("暂停中... 请稍后")
        pauseEle.attr("disabled", true)
        $("#abort").attr("disabled", true)
        // $("#download_model").show()
    }else if (pauseEle.attr("value") === "继续任务"){
        pauseEle.val("暂停任务")
        $("#download_model").hide()
        $("#abort").attr("disabled", false)
    }

}

function killProcess(url) {
    $.get(url)
    var abortButton = document.getElementById("abort");
    abortButton.setAttribute("disabled", "disabled")
    abortButton.setAttribute("value", "正在终止任务... 请稍后")
    // $.ajax({
    //     type: "GET",
    //     url: url,
    //     success: function (){return 0}
    // })
}