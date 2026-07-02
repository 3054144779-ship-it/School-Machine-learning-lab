package com.huahuo.demo.controller;

import com.huahuo.demo.entity.StudentEntity;
import com.huahuo.demo.service.StudentService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

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

    @PostMapping
    public ResponseEntity<StudentEntity> addStudent(@RequestBody StudentEntity student) {
        StudentEntity saved = studentService.saveStudent(student);
        return ResponseEntity.ok(saved);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteStudent(@PathVariable Long id) {
        studentService.deleteStudent(id);
        return ResponseEntity.ok().build();
    }
}