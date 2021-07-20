import io
import base64
import pandas as pd
from django.shortcuts import render
from django.views import View

from .helper_candlestick import get_candlestick_fig


# Create your views here.


def get_b64(fig):
    flike = io.BytesIO()
    fig.savefig(flike)
    b64 = base64.b64encode(flike.getvalue()).decode()
    return b64


class ChartPage(View):
    ObjModel = None
    template_path = 'chart.html'

    def get(self, request, format=None):
        symbol = request.GET.get('symbol', 'ACI')
        if not len(symbol):
            symbol = 'ACI'
        fig = get_candlestick_fig(
            symbol=symbol, data_n=120, resample=False, step='3D'
        )
        bs64 = get_b64(fig)
        cur_df = pd.read_csv('dsebd_current_data.csv')
        symbols = list(cur_df['TRADING_CODE'].values)
        context = {
            'symbol': symbol,
            'image': bs64,
            'symbols': symbols,
            'days': [90, 120, 150],
            'steps': ['1D', '3D', '7D']
        }

        return render(
            request=request,
            template_name=self.template_path,
            context=context,
            status=200
        )


