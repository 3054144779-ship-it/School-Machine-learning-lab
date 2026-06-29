package com.huahuo.demo.entity;

import lombok.Getter;
import lombok.Setter;
import jakarta.persistence.*;

@Entity
@Table(name = "t_student_history")
@Getter
@Setter
public class StudentEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY) // 主键自增
    private Long id;

    @Column(name = "attendance")
    private Double attendance;     // 出勤率

    @Column(name = "homework")
    private Double homework;       // 作业分

    @Column(name = "midterm")
    private Double midterm;        // 期中分

    @Column(name = "participation")
    private Integer participation; // 课堂参与度

    @Column(name = "final_grade")
    private Double finalGrade;     // 期末真实成绩（用于模型训练的历史数据）
}
