from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.contrib import messages

from .models import Prediction
from .ai_models import predict_image, MODELS
from .forms import SignUpForm


MODEL_INFORMATION = {

    'resnet50': {
        'name': 'ResNet50',
        'accuracy': 98.17,
        'precision': 98.14,
        'recall': 98.14,
        'f1': 98.14,
        'auc': 99.76,
        'parameters': '24.11M',
    },

    'mobilenetv2': {
        'name': 'MobileNetV2',
        'accuracy': 96.98,
        'precision': 97.58,
        'recall': 96.26,
        'f1': 96.91,
        'auc': 99.60,
        'parameters': '2.59M',
    },

    'baseline_cnn': {
        'name': 'Baseline CNN',
        'accuracy': 95.01,
        'precision': 93.98,
        'recall': 96.04,
        'f1': 95.00,
        'auc': 98.80,
        'parameters': 'N/A',
    }
}

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024


# Authentication functions

def signup_view(request):

    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Account created successfully.')
            return redirect('dashboard')
    else:
        form = SignUpForm()

    return render(request, 'detector/signup.html', {'form': form})


class CustomLoginView(LoginView):
    template_name = 'detector/login.html'
    redirect_authenticated_user = True


def logout_view(request):
    logout(request)
    return redirect('login')



# APP VIEWS

@login_required
def dashboard(request):

    recent_predictions = (
        Prediction.objects
        .filter(user=request.user)
        .order_by('-created_at')[:5]
    )

    return render(
        request,
        'detector/dashboard.html',
        {
            'recent_predictions': recent_predictions,
            'available_models': list(MODELS.keys()),
        }
    )

# Detect Image function
@login_required
def detect_image(request):

    result = None
    available_models = list(MODELS.keys())

    if request.method == 'POST':

        image = request.FILES.get('image')
        model_name = request.POST.get('model')

        if not image or not model_name:
            result = {'error': 'Please provide both an image and a model.'}

        elif model_name not in MODEL_INFORMATION:
            result = {'error': 'Invalid model selected.'}

        elif model_name not in MODELS:
            result = {
                'error': f'The "{MODEL_INFORMATION[model_name]["name"]}" model is not '
                         f'currently available. Please choose a different model.'
            }

        elif not image.content_type or not image.content_type.startswith('image/'):
            result = {'error': 'Please upload a valid image file.'}

        elif image.size > MAX_IMAGE_SIZE_BYTES:
            result = {'error': 'Image is too large. Maximum size is 10MB.'}

        else:
            try:
                result = predict_image(image, model_name)

                image.seek(0)

                #saving prediction to DB

                prediction = Prediction.objects.create(
                    user=request.user,
                    image=image,
                    model_name=model_name,
                    prediction=(
                        'ai_generated'
                        if result['prediction'] == 'AI-Generated'
                        else 'real'
                    ),
                    confidence=result['confidence'],
                    processing_time=result['processing_time'],
                )

                result['image_url'] = prediction.image.url
                result['model'] = MODEL_INFORMATION[model_name]

            except ValueError as e:
                result = {'error': str(e)}

            except Exception as e:
                result = {'error': f'Something went wrong while processing the image: {e}'}

    return render(
        request,
        'detector/detect.html',
        {
            'result': result,
            'models': MODEL_INFORMATION,
            'available_models': available_models,
        }
    )


@login_required
def model_comparison(request):

    return render(
        request,
        'detector/comparison.html',
        {
            'models': MODEL_INFORMATION,
            'available_models': list(MODELS.keys()),
        }
    )


@login_required
def prediction_history(request):

    predictions = (
        Prediction.objects
        .filter(user=request.user)
        .order_by('-created_at')
    )

    return render(
        request,
        'detector/history.html',
        {
            'predictions': predictions
        }
    )


@login_required
def about(request):

    return render(
        request,
        'detector/about.html'
    )