��    ~                    �     �          "     7     C     T     [     `     e     w     �  #   �     �     �     �     	     	     ,	     >	     P	     b	     t	     �	     �	     �	     �	     �	     �	     �	     
     !
     4
     G
     Z
     m
     �
     �
     �
  
   �
     �
     �
     �
     �
                &     @     R     f     ~     �     �     �     �     �     �     �     �     �     �               7     ?     G     O     W  	   _     i     u     �     �     �     �     �     �     �     �               ,     @     T     h     |     �     �     �     �     �     �     �                    /     ?     O     _     o     }     �     �     �     �     �     �     �                 
   +     6     C     P  	   \     f     t     y     �     �     �     �     �     �  �  �     p     �     �     �     �     �  5   �     -     B     _     {  #   �     �  s   �     F     c  A   y  >   �  [   �  6   V  ?   �    �     �  �   �    �  
  �  6   �  �   4  �   �  r   �  %     Y   B     �     �  ~   �     N  z   d  X   �     8     J  O   S     �     �     �     �     �          '  '   A     i     y     �     �     �     �     �     �             
   9     D     \    y  �   �   �   d!  �   �!  �   �"     �#  +   �#  L   �#     0$     G$  !   U$     w$     �$  b   �$  C  �$     4&  U   O&  5   �&  �   �&  �   �'  �   �(  �   q)    1*    H+     e,  	   ,     �,     �,  ;   �,  P   �,     C-  #   S-  "   w-     �-     �-     �-     �-     .      .     ..     B.  w   _.     �.     �.     /     /     /  !   >/  ?   `/  
   �/     �/     �/     �/  	   �/     �/  E   0  I   J0     �0     �0     �0     �0  !   �0     1   accuracy_metric advance_setting_hide advance_setting_show algo_config algo_para_config and_eo area back choose_model_para choose_model_structure choose_sensitive_attributes conditional_group_fairness_analysis continuous_attribute data_cg_fair_info_1 data_cg_fair_info_4 data_cg_fair_info_5 data_chart_1 data_eval_error_1 data_eval_error_2 data_eval_error_3 data_eval_error_4 data_eval_result_1 data_eval_result_10 data_eval_result_11 data_eval_result_2 data_eval_result_4 data_eval_result_5 data_eval_result_6 data_eval_result_7 data_eval_result_8 data_eval_result_9 data_g_fair_info_1 data_g_fair_info_4 data_g_fair_info_5 data_i_fair_info_1 data_i_fair_info_2 data_info_1  data_info_2  data_intro dataset dataset_duplicate_name dataset_fairness_evaluation dataset_name dataset_selection  discrete_attribute discrimination_preference dont_keep_dataset download_data_temp  download_model_template download_models eval_num fairness_evaluation fairness_metric fairness_range filename_conflict finished_task func1 func2 func3 group group_fairness_analysis individual_fariness_analysis intro_1 intro_2 intro_3 intro_4 intro_5 iter_time label_error legi_feature_info_1 longterm_save_dataset  metrics_score ml_model_eval  mo_optimazor model  model_chart_1 model_def_file_dec model_def_file_demo model_eval_error_1 model_eval_error_2 model_eval_result_1 model_eval_result_2 model_eval_result_3 model_eval_result_4 model_eval_result_5  model_eval_result_6 model_fairness_eval next optimization_obj optimization_process optimization_result platform_name pop_size progress_info_1 progress_info_2 progress_info_3 progress_info_4 progress_info_5 progress_info_6 provided_data running_task select_data_file  select_proper_attributes sens_feature_info_1 sensitive_attributes target_attribute  task_continue task_end task_error1 task_error2 task_error3 task_pause task_pausing task_running task_status task_stop task_stopping unit update_error upload_dataset  upload_init_models upload_ml_model upload_model upload_model_eval yes Project-Id-Version: PROJECT VERSION
Report-Msgid-Bugs-To: EMAIL@ADDRESS
POT-Creation-Date: 2022-04-29 16:54+0800
PO-Revision-Date: 2021-12-27 11:32+0800
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: en_US
Language-Team: en_US <LL@li.org>
Plural-Forms: nplurals=2; plural=(n != 1)
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.9.0
 Accuracy metric advance setting: hide advance setting: show Algorithm configuration Configure algorithm parameters  and Equalized Odds Nanshan District, Shenzhen, Guangdong Province, China Return to start page Choose model parameters file Choose model structure file Select sensitive attributes Conditional group fairness analysis Numerical attribute(s) Conditional group fairness analysis: calculate the group fairness under subgroups divided by legitimate attributes.  ·  Defaut fairness range:  This metric refers to Ratio of group positive label rate to overall positive label rate At least one sensitive attribute must be selected for analysis At least one sensitive attribute and one legitimate attribute must be selected for analysis Legitimate attributes must be non-sensitive attributes At least one legitimate attribute must be selected for analysis The positive label (label={}) rate of "{}" group is lower than the positive label rate of whole dataset. It is recommended to check whether the group is discriminated \ preferenced, and this attribute should be non-sensitive attribute which influences the result. The group partitioned by "{}" After using "{}" as the legitimate attribute to analyse the attribute "{}", no fairness issue was found. Or user should still conduct further analyses on this data, seek input from all stakeholders (especially affected marginalized communities). The positive label (label={}) rate of "{}" group is higher than the positive label rate of whole dataset. It is recommended to check whether the group is discriminated \ preferenced, and this attribute should be non-sensitive attribute which influences the result. No fairness issue was found in group fairness analysis for sensitive attribute "{}", user can try other sensitive attributes. Or user should still conduct further analyses on this data, seek input from all stakeholders (especially affected marginalized communities). No fairness problems were found in the current dataset The radio of positive label rate between the subgroup "{}"="{}" of group "{}"="{}" and the whole group (positive label: "{}"="{}") is low, this group may be discriminated \ preferenced. The radio of positive label rate between the subgroup "{}"="{}" of group "{}"="{}" and the whole group (positive label: "{}"="{}") is high, this group may be discriminated \ preferenced. After analysing the legitimate attribute "{}", the following "{}" individuals may be discriminated \ preferenced:  The analysis result of attribute "{}" Group fairness analysis: calculate the radio of positive rate between groups and overall.  ·  Defaut fairness range:  This metric refers to Individual fairness analysis: prediction should be the same for any two subjects with the exact same non-sensitive attributes. This metric refers to Please click "OK" after selecting the data set. To upload a dataset, click "Upload dataset" after selecting the data file. After confirming the selected dataset and checking the property list, click "Next step". Data introduction Dataset  Upload fail: The name of uploaded dataset is same to the datasets already exsit Dataset fairness evaluation Dataset name Dataset selection Categorical attribute(s) discriminated \ preferenced Do not keep dataset Download dataset template Download model definition file template Download models Evolutionary evaluation  fairness evaluation Fairness metric Fairness range Filename conflict Finished tasks Fairness analysis on data Fairness analysis on ML models Training fairer ML models group "{}" Group fairness analysis Individual fairness analysis This platform provides the function of analyzing the fairness problems existing in machine learning data and models, and alleviating the inequity of models by optimizing models with multi-objective algorithms. Fairness in machine learning generally refers to [1]. People who are identical on non-sensitive attributes should have similar predicted results (or labels). For example, for students with the same grades, regardless of gender, the model should predict the same admission outcome. The predicted results (or labels) between different groups should not differ more than the differences between their non-sensitive characteristics. Typically, there are certain characteristics recognized by law that do not permit discrimination, and in the computer science literature, these characteristics are often considered as "protected" or "sensitive" attributes [2]. FairerML is capable of training Pareto model set considering simultaneously accuracy and one or more fairness metrics with multi-objective optimisation, specific algorithm implementation refers to [3]. Generation number Label can only contain two different values Legitimate attributes is non-sensitive attribute that could influence labels Longterm save datasets metric scores Machine learning model evaluation MOEA optimizer Model  Ratio of group positive label rate between overall and groups divided by this legitimate attribute Please refer to the Pytorch model definition file (.py) and the saved parameters (.pth or .plk) file. Saved parameters file must be exported by "torch.save(model.state_dict(),path)". Model definition file must contain a subclass of torch.nn.Module and a main() with no arguments, this function will return a model instance. Model definition file demo A legitimate attribute must be selected to perform conditional group equity analysis. Legitimate attribute must be non-sensitive attribute. According to "{}", group "{}" may be "{}". It is suggested to consider fairness metrics "{}" as one of the optimization objective or considering whether "{}" influences "{}". If yes, it is suggested to remove it from sensitive attributes list. No fairness issue was found in the analysis of "{}". User should still conduct further analyses, seek input from all stakeholders (especially affected marginalized communities), and actively monitor the model if it's deployed. The radio of positive prediction rate between the subgroup "{}"="{}" of group "{}"="{}" and the whole group (positive label: "{}"="{}") is low, this group may be discriminated \ preferenced. The radio of positive prediction rate between the subgroup "{}"="{}" of group "{}"="{}" and the whole group (positive label: "{}"="{}") is high, this group may be discriminated \ preferenced. After using "{}" as the legitimate attribute to analyse the attribute "{}", no fairness issue was found. User should still conduct further analyses, seek input from all stakeholders (especially affected marginalized communities), and actively monitor the model if it's deployed. The metric score(s) "{}" is low, the groups divided by sensitive attribute "{}" may exsit bias. It is suggested to take these metrics as the optimization objectives. Or considering whether attribute "{}" has influences on "{}". If yes, please remove it from sensitive attributes list. Model fairness evaluation Next step Optimization objectives Optimization process Optimization result for accuracy metric and fairness metric FairerML: Machine Learning Fairness Evaluation and Suggestion Platform (Demo v1) Population size Evolving ... generation {}/{}({}%)  Initializing problem instances ... Initializing population Initializing parameters ... Evolution finished! Task stop! generation{}/{}({}%) Datasets provided by FairerML Running tasks Select dataset file Select legitimate attributes Sensitive attributes cannot influence labels, and some groups in sensitive attributes may be discriminated / preferred. Sensitive attribute(s) Target attribute / Label Continue task Finished Population size must be digital Generation number must be digital Continuous attributes are not supported as sensitive attributes Pause task Task_pausing ... please wait Running Task status Stop task Task_stopping ... please wait Research Institute of Trustworthy Autonomous Systems (RITAS), SUSTech The format of dataset is incorrect, please check by reading template file Upload dataset Upload initialization models Upload machine learning model Please upload model Upload model and evaluation model OK 