package com.huahuo.demo.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 *  NotNull：这个字段前端必须传，不能为空。
 *  Min(0) / @Max(100)：数值必须在 0 到 100 之间。
 *  Valid：挂在 Controller 的参数前，意思是“当这个请求进来时，请 Spring 按照 DTO 上的标签，挨个检查一遍数据”。
 * */
@Data
public class PredictionRequest {
    @NotNull @Min(0) @Max(100)
    private Double attendance;

    @NotNull @Min(0) @Max(100)
    private Double homework;

    @NotNull @Min(0) @Max(100)
    private Double midterm;

    @NotNull @Min(1) @Max(10)
    private Integer participation;
}
