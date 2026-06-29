package com.huahuo.demo.service;

import com.huahuo.demo.entity.StudentEntity;
import com.huahuo.demo.repository.StudentRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class StudentService {

    private final StudentRepository studentRepository;

    public StudentService(StudentRepository studentRepository) {
        this.studentRepository = studentRepository;
    }

    /**
     * 获取所有学生历史数据，供前端分析面板绘制热力图和条形图
     */
    public List<StudentEntity> getAllStudentHistory() {
        return studentRepository.findAll();
    }
}