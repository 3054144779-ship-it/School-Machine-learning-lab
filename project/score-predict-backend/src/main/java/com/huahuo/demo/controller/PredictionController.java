package com.huahuo.demo.controller;

import com.huahuo.demo.dto.PredictionRequest;
import com.huahuo.demo.service.PredictionService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/prediction")
@CrossOrigin(origins = "*")
public class PredictionController {

    private final PredictionService predictionService;

    public PredictionController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @PostMapping("/predict")
    public ResponseEntity<Map<String,Object>> predict(@Valid @RequestBody PredictionRequest request){
        // 调用 Service 层进行预测
        Double finalGrade = predictionService.predictFinalGrade(request);

        // 封装统一的返回格式
        Map<String, Object> response = new HashMap<>();
        response.put("code", 200);
        response.put("message", "预测成功");
        response.put("predictedGrade", finalGrade);

        return ResponseEntity.ok(response);
    }

}
