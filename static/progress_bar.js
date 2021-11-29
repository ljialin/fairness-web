function start_task(pid) {
    // 设置定时器,隔段时间请求一次数据
    var sitv = setInterval(function () {
        // prog_url指向请求进度的url，后面会在flask中设置
        var prog_url = '/task/' + pid + '/show_progress'
        $.getJSON(prog_url, function (res) {
            $('.progress-div').css('visibility', 'visible');
            $('.progress-bar').css('width', res.progress_rate + '%')
                .css('background', 'green')
                .css('text-align', 'center')
                .text("-");
            $('#prog_info').text(res.progress_info);
            var str = ""
            for (let i = 0; i < res.pop.length; i++) {
                str += "<tr><td>" + i +
                    "</td><td>" + res.pop[i][0] +
                    "</td><td>" + res.pop[i][1] + "</td></tr>";
                $("#poptext").html(str);
            }
        });
        get_chart(pid,"http://127.0.0.1:5000/task/" + pid + "/chart")
        if (res.progress_rate === -1){
            clearInterval(sitv);
        }
    }, 1000);

    // 点击事件第一个请求地址，发送请求，后台业务开始执行
    var this_url = '/task/' + pid + '/progress'
    $.getJSON(this_url, function (res) {
        // alert(res.progress_status)
        if (res.progress_status === 12){
            clearInterval(sitv);
            $('.progress-bar').css('background', 'red');
            // setTimeout(function () {
            //     alert('失败了!');
            // }, 1);
        }else if(res.progress_status === 11){
            clearInterval(sitv);
            $('.progress-div').css('visibility', 'visible');
            $('.progress-bar').css('width', '100%')
                .css('background', 'green')
                .css('text-align', 'center')
                .text("-");
            $('#prog_info').text(res.progress_info);
            // setTimeout(function () {
            //     alert('运行成功!');
            // }, 100);
        }else{
            $('.progress-bar').css('width', res.progress_rate + '%')
                .css('background', 'green')
                .css('text-align', 'center')
                .text("-");
            $('#prog_info').text(res.progress_info);
        }
    });
}
