# 该代码是核心程序的独立运行代码，可供了解代码运行方式。

data <- read_csv("C:/Users/Administrator/Desktop/workspace/2025.10.30hitom/hitom7.xlsx")

df <- data |> 
  mutate(
    data_id = ifelse(
      grepl("^[A-Za-z]\\d+", Sort), Sort, NA
    ), .before = 1
  ) |> 
  tidyr::fill(data_id, .direction = "down") |> 
  filter(!is.na(data_id) & !is.na(Ratio))

# 识别单方向突变类型
df <- df %>%
  mutate(
    type = case_when(
      `Left variation` == "WT" & `Left variation type` == "WT" ~ "WT",
      `Left variation` == "WT" ~ `Left variation type`,
      `Left variation type` == "WT" ~ `Left variation`,
      TRUE ~ paste(`Left variation`, `Left variation type`, sep = "\\")
    ), .before = 1
  )


df <- df %>%
  mutate(
    type = case_when(
      `Left variation` == "WT" & `Left variation type` == "WT" ~ "WT",
      `Left variation` == "WT" ~ paste0(`Left variation type`, "(", `Right variation type`, ")"),
      `Left variation type` == "WT" ~ paste0(`Left variation`, "(", `Right variation`, ")"),
      TRUE ~ paste0(
        `Left variation`, "(", `Right variation`, ")", 
        "\\", 
        `Left variation type`, "(", `Right variation type`, ")"
      )
    ), .before = 1
  )

# 剔除置信度太低的数据，这里硬编码扔掉低于25%的信号
df <- df |>
  mutate(
    Ratio_numeric = as.numeric(sub("%", "", Ratio))/100 # 转化格式
  ) |> 
  filter(Ratio_numeric > 0.25)

# 对比数据的比例，标准化（在抛弃了弱信号的基础上算新比例）
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
      n_samples > 2 ~ "ERROR, too many signals", # 信号太多时不正常，单独看"error_too_many_samples"
      n_samples == 1 ~ type, # "single_sample"
      n_samples == 2 ~ {#"two_samples"
        # 获取两个样本的ratio和type
        
        ratios <- df$std_ratio
        
        types <- df$type
        ratio1 <- ratios[1]
        ratio2 <- ratios[2]
        type1 <- types[1]
        type2 <- types[2]
        
        # 判断是否在1.5~0.6范围内
        ratio_ratio <- ratio1 / ratio2
        if (ratio_ratio >= 0.6 && ratio_ratio <= 1.5) {
          # 在范围内，输出两个type，用"/"间隔
          paste(type1, type2, sep = "/")
        } else {
          # 不在范围内，输出更大的数据的type
          type1
        }
      }
    ), .before = 1
  ) %>%
  ungroup()

# 输出
result <- df|> 
  distinct(data_id, final_type)

