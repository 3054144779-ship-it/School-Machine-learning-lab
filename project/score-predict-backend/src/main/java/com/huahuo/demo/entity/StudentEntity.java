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
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "student_name")
    private String studentName;          // 学生姓名

    @Column(name = "interaction")
    private Double interaction;          // 线下_互动

    @Column(name = "comprehensive_regular")
    private Double comprehensiveRegular; // 综合_平时成绩

    @Column(name = "final_total")
    private Double finalTotal;           // 期末总成绩

    @Column(name = "regular_score")
    private Double regularScore;         // 平时成绩

    @Column(name = "final_score")
    private Double finalScore;           // 期末成绩

    @Column(name = "online_total")
    private Double onlineTotal;          // 线上总成绩（目标变量）
}
