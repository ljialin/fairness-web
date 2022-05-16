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
    if (pauseEle.attr("value") === _("task_pause")){
        pauseEle.val(_("task_pausing"))
        pauseEle.attr("disabled", true)
        $("#abort").attr("disabled", true)
        // $("#download_model").show()
    }else if (pauseEle.attr("value") === _("task_continue")){
        pauseEle.val(_("task_pause"))
        $("#download_model").hide()
        $("#abort").attr("disabled", false)
    }

}

function killProcess(url) {
    $.get(url)
    var abortButton = document.getElementById("abort");
    abortButton.setAttribute("disabled", "disabled")
    abortButton.setAttribute("value", _("task_stopping"))
    // $.ajax({
    //     type: "GET",
    //     url: url,
    //     success: function (){return 0}
    // })
}

function selectFairMetric(obj, id, url){
    var index = obj.selectedIndex;
    $.get(url + "select_fair/" + index);
    get_chart(id, url + "chart");
}

function showAdvance(){
    var advance = $("#advance");
    var button = $("#advance_setting")
    if (advance.is(":hidden")){
        advance.show();
        button.val(_("advance_setting_hide"))
    }else{
        advance.hide();
        button.val(_("advance_setting_show"))
    }
}

function changeThreshold(url){
    var threshold = $('#threshold').val()
    $.getJSON(url + threshold, function (res) {
        alert(res.err);
    });
}
