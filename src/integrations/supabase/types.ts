export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "14.5"
  }
  public: {
    Tables: {
      conversations: {
        Row: {
          category: string
          created_at: string
          id: string
          model_id: string
          title: string
          updated_at: string
          user_id: string
        }
        Insert: {
          category?: string
          created_at?: string
          id?: string
          model_id?: string
          title?: string
          updated_at?: string
          user_id: string
        }
        Update: {
          category?: string
          created_at?: string
          id?: string
          model_id?: string
          title?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      credit_events: {
        Row: {
          amount: number
          balance_after: number
          created_at: string
          event_type: string
          id: string
          reason: string | null
          user_id: string
        }
        Insert: {
          amount: number
          balance_after: number
          created_at?: string
          event_type: string
          id?: string
          reason?: string | null
          user_id: string
        }
        Update: {
          amount?: number
          balance_after?: number
          created_at?: string
          event_type?: string
          id?: string
          reason?: string | null
          user_id?: string
        }
        Relationships: []
      }
      guest_usage: {
        Row: {
          created_at: string
          credits_used: number
          guest_id: string
          updated_at: string
        }
        Insert: {
          created_at?: string
          credits_used?: number
          guest_id: string
          updated_at?: string
        }
        Update: {
          created_at?: string
          credits_used?: number
          guest_id?: string
          updated_at?: string
        }
        Relationships: []
      }
      messages: {
        Row: {
          conversation_id: string
          created_at: string
          id: string
          parts: Json
          role: string
          user_id: string
        }
        Insert: {
          conversation_id: string
          created_at?: string
          id?: string
          parts?: Json
          role: string
          user_id: string
        }
        Update: {
          conversation_id?: string
          created_at?: string
          id?: string
          parts?: Json
          role?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "messages_conversation_id_fkey"
            columns: ["conversation_id"]
            isOneToOne: false
            referencedRelation: "conversations"
            referencedColumns: ["id"]
          },
        ]
      }
      profiles: {
        Row: {
          avatar_url: string | null
          created_at: string
          display_name: string | null
          updated_at: string
          user_id: string
        }
        Insert: {
          avatar_url?: string | null
          created_at?: string
          display_name?: string | null
          updated_at?: string
          user_id: string
        }
        Update: {
          avatar_url?: string | null
          created_at?: string
          display_name?: string | null
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      user_api_keys: {
        Row: {
          nvidia_api_key: string | null
          updated_at: string
          user_id: string
        }
        Insert: {
          nvidia_api_key?: string | null
          updated_at?: string
          user_id: string
        }
        Update: {
          nvidia_api_key?: string | null
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      user_credits: {
        Row: {
          created_at: string
          credits: number
          total_used: number
          updated_at: string
          user_id: string
        }
        Insert: {
          created_at?: string
          credits?: number
          total_used?: number
          updated_at?: string
          user_id: string
        }
        Update: {
          created_at?: string
          credits?: number
          total_used?: number
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      user_models: {
        Row: {
          api_key_ciphertext: string | null
          base_url: string
          category: string
          created_at: string
          enabled: boolean
          id: string
          label: string
          model_id: string
          provider: string
          updated_at: string
          user_id: string
        }
        Insert: {
          api_key_ciphertext?: string | null
          base_url: string
          category?: string
          created_at?: string
          enabled?: boolean
          id?: string
          label: string
          model_id: string
          provider: string
          updated_at?: string
          user_id: string
        }
        Update: {
          api_key_ciphertext?: string | null
          base_url?: string
          category?: string
          created_at?: string
          enabled?: boolean
          id?: string
          label?: string
          model_id?: string
          provider?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      consume_credit: { Args: { _user_id: string }; Returns: number }
      consume_guest_credit: {
        Args: { _guest_id: string; _limit?: number }
        Returns: number
      }
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {},
  },
} as const
