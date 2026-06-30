package com.huahuo.demo.controller;

import com.huahuo.demo.dto.PredictionRequest;
import com.huahuo.demo.service.PredictionService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

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
    public ResponseEntity<Map<String, Object>> predict(@Valid @RequestBody PredictionRequest request) {
        Map<String, Object> result = predictionService.predict(request);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/features")
    public ResponseEntity<Map<String, Object>> features() {
        Map<String, Object> result = predictionService.getFeatures();
        return ResponseEntity.ok(result);
    }

    @GetMapping("/analysis")
    public ResponseEntity<Map<String, Object>> analysis() {
        Map<String, Object> result = predictionService.getAnalysis();
        return ResponseEntity.ok(result);
    }

    @GetMapping("/tree")
    public ResponseEntity<Map<String, Object>> tree() {
        Map<String, Object> result = predictionService.getTree();
        return ResponseEntity.ok(result);
    }
}
