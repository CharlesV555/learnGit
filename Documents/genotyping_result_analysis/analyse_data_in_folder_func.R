library(tidyverse)
library(openxlsx)

source("summarize_data_in_folder_func.R")
source("genptype_marking_func.R")

#-------------------------------------------------------------------------------
# usage: put all table,downloaded from Hi-TOM website, in the same folder based on 
# order(you should have marked each plate).
# Then write down the path to the folder into below path, run the script, then you get a xlxs outcome
# in the working direction.
# 介绍：这套流程可以将Hi-TOM网站返回文件中的Sequence文件从展示信号读数的形式转换
#       为紧凑的基因型表格。你可以设置判断标准。
# 预期：你的一个板子上测的数据返回了几个.xlsx文件，你希望只得到这个板子中样本发
#       生突变的基因型。
# 用法：将你从Hi-TOM网站上下载文件中的“*Sequence.xls”文件放在同一个文件夹中（最
#       好是同一板的数据放在同一个文件夹，这样输出的excel表格中只包括一个板子的
#       信息）

# 结果输出路径，比如"C:/Users/Administrator/Desktop/workspace/2025.10.30hitom"
output_path = "C:/Users/Administrator/Desktop/workspace/2025.10.30hitom"

# 数据所在文件夹路径，
# 比如"C:/Users/Administrator/Desktop/workspace/2025.10.30hitom/data/h7"
path_to_your_data_folder = "C:/Users/Administrator/Desktop/workspace/2025.10.30hitom/data/h7"

# 结果名字
name_the_result_file = "hitom7"

# 运行流程。
summarize_data_in_folder(path_to_your_data_folder, 
                         name_the_result_file, 
                         output_path,
                         min_threshold=0.25, # 信号强度弱于这个值的会被过滤
                         compare_index=1.5 # 过滤弱信号并重新标准化后如果存在两个信号，信号比值小于此比例的认为是杂合子
                         )

# 接下来需要靠你自己整理表格了，祝君好运。

# 如果指定了output_path，去那里找文件；
# 如果没有指定output_path，下一行会告诉你文件在哪：
getwd() 
