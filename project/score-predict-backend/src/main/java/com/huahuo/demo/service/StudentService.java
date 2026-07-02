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

    public List<StudentEntity> getAllStudentHistory() {
        return studentRepository.findAll();
    }

    public StudentEntity saveStudent(StudentEntity student) {
        return studentRepository.save(student);
    }

    public void deleteStudent(Long id) {
        studentRepository.deleteById(id);
    }
}