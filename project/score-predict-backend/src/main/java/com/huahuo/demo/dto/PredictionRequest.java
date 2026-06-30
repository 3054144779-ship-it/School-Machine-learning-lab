package com.huahuo.demo.dto;

import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.util.List;

@Data
public class PredictionRequest {
    @NotNull
    @NotEmpty
    private List<Double> features;
}
