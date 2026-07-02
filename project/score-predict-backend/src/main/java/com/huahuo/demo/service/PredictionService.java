package com.huahuo.demo.service;

import com.huahuo.demo.dto.PredictionRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

import java.util.LinkedHashMap;
import java.util.Map;

@Service
public class PredictionService {

    private static final Logger log = LoggerFactory.getLogger(PredictionService.class);

    private final RestTemplate restTemplate;

    @Value("${python.api.url}")
    private String pythonApiUrl;

    public PredictionService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Map<String, Object> predict(PredictionRequest request) {
        log.info("接收到预测请求：{}", request);

        Map<String, Object> pythonReq = new LinkedHashMap<>();
        pythonReq.put("features", request.getFeatures());

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(pythonReq, headers);

        String url = pythonApiUrl + "/api/predict";
        log.info("调用 Python API: {}", url);

        try {
            Map<String, Object> pythonResp = restTemplate.postForObject(url, entity, Map.class);
            log.info("Python API 响应: {}", pythonResp);
            return pythonResp;
        } catch (RestClientException e) {
            log.error("调用 Python API 失败: {}", e.getMessage());
            // 尝试从 Python 响应中提取错误信息
            String detail = e.getMessage();
            try {
                // 从异常消息中提取 JSON body
                String bodyStr = detail.substring(detail.indexOf("\"detail\":\"") + 10);
                bodyStr = bodyStr.substring(0, bodyStr.indexOf("\""));
                detail = bodyStr;
            } catch (Exception ignored) {}
            return Map.of("code", 500, "message", "预测失败: " + detail);
        }
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> getFeatures() {
        String url = pythonApiUrl + "/api/features";
        try {
            return restTemplate.getForObject(url, Map.class);
        } catch (Exception e) {
            log.warn("获取特征信息失败: {}", e.getMessage());
            return Map.of("code", 500, "message", "无法获取特征信息");
        }
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> getAnalysis() {
        String url = pythonApiUrl + "/api/analysis";
        log.info("调用 Python API: {}", url);
        try {
            return restTemplate.getForObject(url, Map.class);
        } catch (Exception e) {
            log.error("获取分析数据失败: {}", e.getMessage());
            return Map.of("code", 500, "message", "分析服务暂时不可用");
        }
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> getTree() {
        String url = pythonApiUrl + "/api/tree";
        log.info("调用 Python API: {}", url);
        try {
            return restTemplate.getForObject(url, Map.class);
        } catch (Exception e) {
            log.error("获取决策树数据失败: {}", e.getMessage());
            return Map.of("code", 500, "message", "决策树服务暂时不可用");
        }
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> getTrainOptions(String source) {
        String url = pythonApiUrl + "/api/train/options?source=" + source;
        log.info("调用 Python API: {}", url);
        try {
            return restTemplate.getForObject(url, Map.class);
        } catch (Exception e) {
            log.error("获取训练选项失败: {}", e.getMessage());
            return Map.of("code", 500, "message", "无法获取训练选项");
        }
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> train(Map<String, Object> config) {
        String url = pythonApiUrl + "/api/train";
        log.info("调用 Python API (训练): {}", url);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(config, headers);

        try {
            return restTemplate.postForObject(url, entity, Map.class);
        } catch (Exception e) {
            log.error("调用训练 API 失败: {}", e.getMessage());
            return Map.of("code", 500, "message", "训练服务暂时不可用: " + e.getMessage());
        }
    }
}
