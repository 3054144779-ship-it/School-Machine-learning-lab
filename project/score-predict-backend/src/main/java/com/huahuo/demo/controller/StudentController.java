package com.huahuo.demo.controller;

import com.huahuo.demo.entity.StudentEntity;
import com.huahuo.demo.service.StudentService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/students")
@CrossOrigin(origins = "*")
public class StudentController {

    private final StudentService studentService;

    public StudentController(StudentService studentService) {
        this.studentService = studentService;
    }

    @GetMapping("/history")
    public ResponseEntity<List<StudentEntity>> getStudentHistory() {
        List<StudentEntity> historyData = studentService.getAllStudentHistory();
        return ResponseEntity.ok(historyData);
    }
}