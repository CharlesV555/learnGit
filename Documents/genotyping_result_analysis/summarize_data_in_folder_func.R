summarize_data_in_folder <- function(folder_path, output_filename, output_path = NULL
                                     , min_threshold, compare_index){
  
  # 获取所有xls文件
  xls_files <- list.files(path = folder_path, 
                          pattern = "\\.xls$|\\.xlsx$", 
                          full.names = TRUE)
  
  # 处理所有表格，每个表格返回两列结果, 这里调用了process_single_file函数
  data_list <- map(xls_files, ~process_single_file(.x, min_threshold, compare_index)) %>% compact()
  
  # 全外连接（保留所有data_id）
  merged_full <- Reduce(function(x, y) {
    merge(x, y, by = "data_id", all = TRUE)
  }, data_list)
  
  # 确保文件名以.xlsx结尾
  if (!grepl("\\.xlsx$", output_filename)) {
    output_filename <- paste0(output_filename, ".xlsx")
  }
  
  if (is.null(output_path)) {
    output_file <- output_filename  # 工作目录
  } else {
    output_file <- file.path(output_path, output_filename)
  }
  
  # 保存为Excel文件
  write.xlsx(merged_full, output_file)
}

# 定义处理单个文件的函数
process_single_file <- function(file_path, min_threshold, compare_index) {
  tryCatch({
    # 读取Excel文件并处理数据，生成两列表格
    result <- genotype_marking(file_path, min_threshold=min_threshold, compare_index=compare_index, name=NA)
    return(result)
  }, error = function(e) {
    warning(paste("处理文件时出错:", file_path, "-", e$message))
    return(NULL)
  })
}

# 仅在测试该代码时调用
# test <- summarize_data_in_folder("C:/Users/Administrator/Desktop/workspace/data/3", "3板")
