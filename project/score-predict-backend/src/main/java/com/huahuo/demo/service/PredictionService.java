package com.huahuo.demo.service;

import com.huahuo.demo.dto.PredictionRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class PredictionService {

    private static final Logger log = LoggerFactory.getLogger(PredictionService.class);

    public Double predictFinalGrade(PredictionRequest request) {
        log.info("接收到预测请求：{}", request);

        // 临时写一个简单的加权算法占位，证明链路打通
        double mockResult = (request.getAttendance() * 0.3) +
                (request.getHomework() * 0.3) +
                (request.getMidterm() * 0.3) +
                (request.getParticipation() * 1.0);

        // 保证分数在 0-100 之间
        return Math.min(Math.max(mockResult, 0.0), 100.0);
    }

}
