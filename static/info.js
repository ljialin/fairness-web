function sens_feature_info(){
    alert("敏感特征指不应影响标签特征值，且可能存在歧视的特征")
}

function legi_feature_info(){
    alert("正当特征指应当影响标签特征值的非敏感特征")
}

function data_g_fair_info(){
    alert(
        "群体公平分析：计算每个群体的正面标签率与总体的正面标签率的比值\n" +
        " ·  若指标值过低，则表明该群体受到歧视\n" +
        " ·  若指标值过低，则表明该群体受到优待\n" +
        " ·  默认公平范围为：[θ, 1/θ]，θ = 0.8)\n\n" +
        "该指标主要参考了Statistical Parity:\n" +
        " ·  Corbett-Davies S, Pierson E, Feller A, et al. Algorithmic decision making and the cost of fairness[C]//Proceedings of the 23rd acm sigkdd international conference on knowledge discovery and data mining. 2017: 797-806."
    )
}

function data_cg_fair_info(){
    alert(
        "条件性群体公平分析：在正当特征值相同的前提下计算每个子群体的公平性：\n" +
        " ·  若指标值过低，则表明该子群体受到歧视\n" +
        " ·  若指标值过低，则表明该子群体受到优待\n" +
        " ·  默认公平范围为：[θ, 1/θ]，θ = 0.8)\n\n" +
        "该指标主要参考Conditional Statistical Parity:\n" +
        " ·  Corbett-Davies S, Pierson E, Feller A, et al. Algorithmic decision making and the cost of fairness[C]//Proceedings of the 23rd acm sigkdd international conference on knowledge discovery and data mining. 2017: 797-806."
    )
}

function data_i_fair_info(){
    alert(
        "个体公平分析：计算每个个体的标签值是否与其大部分近似个体的标值都相同\n" +
        "该指标主要参考Causal Discrimination:\n" +
        " ·  Galhotra S, Brun Y, Meliou A. Fairness testing: testing software for discrimination[C]//Proceedings of the 2017 11th Joint Meeting on Foundations of Software Engineering. 2017: 498-510."
    )
}
