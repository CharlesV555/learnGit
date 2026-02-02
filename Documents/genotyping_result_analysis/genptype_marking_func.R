genotype_marking <- function(data_path, min_threshold=0.25, compare_index=1.5, name = NA){
  
 #data_path <- "C:/Users/Administrator/Desktop/workspace/data/3/hitom3-CAF1K-Sequence.xls"
 #min_threshold=0.25
 #compare_index=1.5
 #name = NA
  
  data <- readr::read_tsv(data_path)
  
  # 将多个信号的ID全部统一成对应sample ID
  df <- data |> 
    mutate(
      data_id = ifelse(
        grepl("^[A-Za-z]\\d+", data$Sort), data$Sort, NA
      ), .before = 1
    ) |> 
    tidyr::fill(data_id, .direction = "down") |> 
    filter(!is.na(data_id) & !is.na(Ratio))
  
  # 识别单方向突变类型
  df <- df %>%
    mutate(
      type = case_when(
        `Left variation` == "WT" & `Left variation type` == "WT" ~ "WT", # pure WT
        `Left variation` == "WT" ~ paste0(`Left variation type`, "(", `Right variation type`, ")"), #两个有一个不是WT，有一个是，则以不一样的为准
        `Left variation type` == "WT" ~ paste0(`Left variation`, "(", `Right variation`, ")"),
        ## 两个都不是WT
        `Left variation` == `Left variation type` & `Right variation` == `Right variation type` ~ paste0(`Left variation`, "(", `Right variation`, ")"), # 两个一样，后缀也一样
        TRUE ~ paste0( # 两个不一样
          `Left variation`, "(", `Right variation`, ")", 
          "\\", 
          `Left variation type`, "(", `Right variation type`, ")"
        )
      ), .before = 1
    )
  
  # 剔除置信度太低的数据
  df <- df |>
    mutate(
      Ratio_numeric = as.numeric(sub("%", "", Ratio)) / 100 # 转化格式
    ) |> 
    filter(Ratio_numeric > min_threshold) # default = 0.25
  
  # 对比数据的比例，标准化
  df <- df |> 
    group_by(data_id) |> 
    mutate(
      std_ratio = Ratio_numeric/sum(Ratio_numeric), .before = 1
    ) |> 
    
    # 检查样本数量，存储在status列
    mutate(
      n_samples = n(),
      .before = 1
    )
  
  # final type determine, 这里涉及到杂合、纯合判定
df <- df |>
      group_by(data_id) |> 
      mutate(
        final_type = case_when(
          n() > 2 ~ "ERROR-too many signals",  # 样本量超过2，标记为错误
          n() == 1 ~ type,    # 单个样本，直接使用其type
          n() == 2 ~ {        # 两个样本的情况
            # 提取两个样本的ratio和type（确保无NA）
            ratio1 <- std_ratio[1]
            ratio2 <- std_ratio[2]
            type1 <- type[1]
            type2 <- type[2]
            
            
            
            current_group <- data_id[1]
            
            # 检查是否有NA值
            if (any(is.na(c(ratio1, ratio2, type1, type2)))) {
              "ERROR_NA_VALUES"
            } else {
              # conventionally we put "WT" behind
              if(type1 == "WT" & type2 != "WT"){
                temp = type2
                type2 = type1
                type1 = temp
                
                temp2 = ratio2
                ratio2 = ratio1
                ratio1 = temp2
              }
              
              # 计算ratio的比例（避免除以0）
              ratio_ratio <- ifelse(ratio2 == 0, Inf, ratio1 / ratio2)
              # 判断是否在合理范围内（假设compare_index已定义，如1.5）
              if (ratio_ratio >= 1/compare_index && ratio_ratio <= compare_index) {
                paste(type1, type2, sep = "/")  # 范围内，合并类型
              } else {
                # 范围外，取ratio较大的类型（避免比较NA）
                if (type2 == "WT"){
                  paste0(type1, "/", type2, "?")
                }else{
                  if (ratio1 >= ratio2) paste0(type1, "/", type2, "?") else paste0(type2, "/", type1, "?")
                }
              }
            }
          },
          .default = "error_in_type_determination"  # 其他情况
        ), .before = 1
      ) |>
      ungroup()
  
  
    file_name <- basename(data_path)
    # 使用正则表达式提取CAF1[Letter]部分
    name <- str_extract(file_name, "CAF1[A-Z]")
    
    # 如果没有匹配到，使用默认列名
    if (is.na(name)) {
      name <- "CAF1_unknown"
      warning(paste("无法从文件名提取CAF1[Letter]:", file_name))
    }
  
  
  # 输出
  df|> 
    distinct(data_id, final_type) |> 
    rename({{name}} := final_type)
}



# test data
# result <- genotype_marking("C:/Users/Administrator/Desktop/workspace/data/7/hitom7-CAF1J-Sequence.xls", 
#                            min_threshold=0.25, compare_index=1.5, name =NA
# )
# View(result)

