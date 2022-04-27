function start_task(pid) {
    // 设置定时器,隔段时间请求一次数据
    var sitv = setInterval(function () {
        // prog_url指向请求进度的url，后面会在flask中设置
        var prog_url = '/task/' + pid + '/show_progress'
        $.getJSON(prog_url, function (res) {
            $('.progress-div').css('visibility', 'visible');
            $('.progress-bar').css('width', res.progress_rate + '%')
                .css('background', '#339999')
                .css('text-align', 'center')
                .text("-");
            $('#prog_info').text(res.progress_info);
            var str = ""
            for (let i = 0; i < res.pop1.length; i++) { //刷新表格
                str += "<tr><td><font color=\"#336699\">" + (i+1);
                for (let j = 0; j < res.pop1[i].length; j++){
                    str += "</font></td><td><font color=\"#336699\">" + res.pop1[i][j];
                }
                str += "</font></td></tr>";
                // str += "<tr><td><font color=\"#336699\">" + (i+1) +
                //     "</font></td><td><font color=\"#336699\">" + res.pop1[i][0] +
                //     "</font></td><td><font color=\"#336699\">" + res.pop1[i][1] + "</font></td></tr>";
                $("#poptext").html(str);
            }
            for (let i = res.pop1.length; i < res.pop1.length + res.pop2.length; i++) { //刷新表格
                str += "<tr><td><font color=\"#696969\">" + (i+1);
                for (let j = 0; j < res.pop2[i-res.pop1.length].length; j++){
                    str += "</font></td><td><font color=\"#696969\">" + res.pop2[i-res.pop1.length][j]
                }
                str += "</font></td></tr>";
                // str += "<tr><td><font color=\"#696969\">" + (i+1) +
                //     "</font></td><td><font color=\"#696969\">" + res.pop2[i-res.pop1.length][0] +
                //     "</font></td><td><font color=\"#696969\">" + res.pop2[i-res.pop1.length][1] + "</font></td></tr>";
                $("#poptext").html(str);
            }
            if (res.progress_status === 15) { //已经暂停了
                var pauseEle = $("#pause");
                pauseEle.attr("disabled", false);
                pauseEle.val(_("task_continue"));
                $("#download_model").show()
            }
        });
        get_chart(pid,"http://127.0.0.1:5000/task/" + pid + "/chart")
        if (res.progress_rate === -1){
            setTimeout(clearInterval(sitv), 1000);
        }
    }, 1000);

    // 点击事件第一个请求地址，发送请求，后台业务开始执行
    var this_url = '/task/' + pid + '/progress'
    $.getJSON(this_url, function (res) {
        // alert(res.progress_status)
        if (res.progress_status === 12){ //error
            clearInterval(sitv);
            $('.progress-bar').css('background', 'red');
            // setTimeout(function () {
            //     alert('失败了!');
            // }, 1);
        }else if(res.progress_status === 11){ //finish
            clearInterval(sitv);
            $('.progress-div').css('visibility', 'visible');
            $('.progress-bar').css('width', '100%')
                .css('background', '#339999')
                .css('text-align', 'center')
                .text(".");
            $('#prog_info').text(res.progress_info);
            $('#abort').hide();
            $('#pause').hide();
            $('#download_model').show();
            // setTimeout(function () {
            //     alert('运行成功!');
            // }, 100);
        }else{ //other
            $('.progress-bar').css('width', res.progress_rate + '%')
                .css('background', '#339999')
                .css('text-align', 'center')
                .text("-");
            $('#prog_info').text(res.progress_info);
        }
    });
}
