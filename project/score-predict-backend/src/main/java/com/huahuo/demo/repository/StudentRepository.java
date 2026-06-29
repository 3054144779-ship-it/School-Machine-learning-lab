package com.huahuo.demo.repository;

import com.huahuo.demo.entity.StudentEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface StudentRepository extends JpaRepository<StudentEntity, Long> {
    // 基础的 findAll(), save() 等方法已经被自动实现
}
