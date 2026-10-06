from django.db import models
from django.contrib.auth.models import User


class Prediction(models.Model):

    MODEL_CHOICES = [
        ('resnet50', 'ResNet50'),
        ('mobilenetv2', 'MobileNetV2'),
        ('baseline_cnn', 'Baseline CNN'),
    ]

    RESULT_CHOICES = [
        ('real', 'Real'),
        ('ai_generated', 'AI-Generated'),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='predictions'
    )

    image = models.ImageField(
        upload_to='uploads/'
    )

    model_name = models.CharField(
        max_length=50,
        choices=MODEL_CHOICES
    )

    prediction = models.CharField(
        max_length=30,
        choices=RESULT_CHOICES
    )

    confidence = models.FloatField()

    processing_time = models.FloatField(
        null=True,
        blank=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return (
            f"{self.user.username} - "
            f"{self.model_name} - "
            f"{self.prediction}"
        )