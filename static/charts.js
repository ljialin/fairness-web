function get_chart(id, path) {
    const chart = echarts.init(document.getElementById(id), 'white', {renderer: 'canvas'});
    $.ajax({
        type: "GET",
        url: "http://127.0.0.1:5000" + path,
        dataType: 'json',
        success: function (result) {
            chart.setOption(result);
        }
    });
}

function render_chart(id, chart_json) {
    const chart = echarts.init(
        document.getElementById(id),
        'white', {renderer: 'canvas'}
    );
    chart.setOption(chart_json);
    // $.ajax({
    //     type: "GET",
    //     url: "http://127.0.0.1:5000" + path,
    //     dataType: 'json',
    //     success: function (result) {
    //         chart.setOption(result);
    //     }
    // });
}
