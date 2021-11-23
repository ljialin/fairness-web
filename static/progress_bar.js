function start_task(pid) {
    // 设置定时器,隔段时间请求一次数据
    var sitv = setInterval(function () {
        // prog_url指向请求进度的url，后面会在flask中设置
        var prog_url = '/task/' + pid + '/show_progress'
        $.getJSON(prog_url, function (num_progress) {
            $('.progress-div').css('visibility', 'visible');
            $('.progress-bar').css('width', num_progress.res)
                .css('background', 'green')
                .css('text-align', 'center')
                .text(num_progress.res);
        });
    }, 2000);

    // 点击事件第一个请求地址，发送请求，后台业务开始执行
    var this_url = '/task/' + pid + '/progress'
    $.getJSON(this_url, function (res) {
        // 清楚定时器
        clearInterval(sitv);
        if (res.res != null) {
            $('.progress-bar').css('width', '100%')
                .text('100%');
            setTimeout(function () {
                alert('运行成功!');
            }, 100);
        } else {
            $('.progress-bar').css('background', 'red');
            setTimeout(function () {
                alert('失败了!');
            }, 1);
        }
    });
}
